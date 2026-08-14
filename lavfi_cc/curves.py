"""``vf_curves.c``'s table construction, in the precision upstream uses.

``curves`` is the one filter of this group that needs no IR extension.  Its
slice functions are ``dst[x + r] = graph[R][src[x + r]]`` and nothing else: one
``1 << depth``-entry table per colour channel, alpha copied.  Everything
interesting happens once, in ``config_input``, where the key points become a
curve.  So the lowering is the same depth-sized LUT as ``lutrgb``'s, and this
module is the part that has to be exact.

Two interpolators produce those tables, and both run in ``double``: a natural
cubic spline through the key points, or PCHIP, which is monotonic between them.
Every operation is a +, -, *, or / that IEEE-754 specifies exactly, so there is
no libm result to distrust here -- but there is still a compiler choice.  A
``double`` multiply-add contracts into one rounding just as a ``float`` one
does, and the polynomial evaluations below are full of them.

So the table is built twice, once with every multiply-add fused and once with
none of them.  Both are cheap at this size.  When they agree -- which is the
usual case -- the table is the same on every host and the lowering needs to
know nothing about this one.  When they do not, the host decides, because
upstream's own bytes differ between hosts there; see :func:`build_table` for
the ordinary curve that provokes it.  The tables are part of the IR, so a
curve that lands on a boundary gets a different plan hash on the two kinds of
host without any further bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import functools
import math
import re


class CurveError(ValueError):
    """A curve specification is outside the accepted subset, or is refused."""


#: The 8-bit table upstream builds: 256 entries indexed by the input byte.
#:
#: Above eight bits ``config_input`` allocates ``1 << depth`` entries instead
#: and passes the depth to both interpolators as ``nbits``, so every function
#: here takes the size rather than reading this.
LUT_SIZE = 256
#: ``lut_size - 1``: what upstream calls ``scale``, the white value.
SCALE = LUT_SIZE - 1

Point = tuple[float, float]

_MATH_FMA = getattr(math, "fma", None)


def _fused(left: float, right: float, addend: float) -> float:
    """One correctly rounded ``double`` fused multiply-add.

    ``math.fma`` only exists from Python 3.13 and this has to run on 3.12 too,
    so there is a fallback that forms the product and sum exactly as rationals
    and rounds once.  ``int.__truediv__`` is correctly rounded, which is what
    makes that final conversion the single rounding a hardware ``fma``
    performs, so the two agree bit for bit -- the fast path only matters
    because a 16-bit table is 65536 entries rather than 256.
    """

    if _MATH_FMA is not None:
        return _MATH_FMA(left, right, addend)  # type: ignore[no-any-return]
    if not (math.isfinite(left) and math.isfinite(right) and math.isfinite(addend)):
        return left * right + addend
    exact = Fraction(left) * Fraction(right) + Fraction(addend)
    return exact.numerator / exact.denominator


@dataclass(frozen=True)
class Arithmetic:
    """The two ways a compiler may evaluate ``a * b`` next to an add.

    ``fused`` is what Clang emits at its default ``-ffp-contract=on`` on a
    target with the instruction; the other is what the written expression says.
    The interpolators below are transcribed once and evaluated under both.
    """

    fuse: bool

    def multiply_add(self, left: float, right: float, addend: float) -> float:
        """``left * right + addend``."""

        if self.fuse:
            return _fused(left, right, addend)
        return left * right + addend

    def multiply_subtract(self, left: float, right: float, subtrahend: float) -> float:
        """``left * right - subtrahend``, which Clang fuses as ``fma(a, b, -c)``."""

        if self.fuse:
            return _fused(left, right, -subtrahend)
        return left * right - subtrahend

    def subtract_product(self, minuend: float, left: float, right: float) -> float:
        """``minuend - left * right``, which Clang fuses as ``fma(-a, b, c)``."""

        if self.fuse:
            return _fused(-left, right, minuend)
        return minuend - left * right


ARITHMETICS = (Arithmetic(True), Arithmetic(False))


def _clip(value: float, scale: int = SCALE) -> int:
    """``CLIP(v)``: ``av_clip_uint8`` at eight bits, ``av_clip_uintp2_c`` above.

    Both are a truncating conversion followed by a clamp to ``[0, scale]``,
    where ``scale`` is ``lut_size - 1``.
    """

    if not value > 0.0:
        return 0
    if value >= float(scale):
        return scale
    return int(value)


def _table_index(value: float, scale: int = SCALE) -> int:
    """``(int)(point->x * scale)``, which truncates toward zero."""

    return int(value * scale)


#: A plain decimal literal.  ``av_strtod`` additionally accepts SI postfixes,
#: so ``0.5k`` means 500 upstream; those are refused rather than honoured, for
#: the same reason ``lutyuv=r=`` is.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def parse_points(text: str, option: str, scale: int = SCALE) -> tuple[Point, ...]:
    """Reproduce ``parse_points_str``, refusing what it would misread.

    Upstream reads a number, steps over exactly one character, reads another,
    and steps again -- so the separators are positional rather than meaningful
    and ``0/0 1/1`` and ``0-0*1-1`` parse the same.  That control flow is kept;
    what is not kept is ``av_strtod`` accepting anything but a decimal literal.

    ``scale`` is ``lut_size - 1``, which decides how close two key points may
    be: two that share a table entry are a configuration failure, and a deeper
    table has finer entries, so a pair upstream rejects at eight bits it can
    accept at ten.
    """

    points: list[Point] = []
    position = 0
    length = len(text)
    while position < length:
        coordinates: list[float] = []
        for _ in range(2):
            while position < length and text[position].isspace():
                position += 1
            match = _NUMBER.match(text, position)
            if match is None or position >= length:
                raise CurveError(
                    f"{option} must be key points spelled 'x/y x/y ...' with "
                    f"plain decimal coordinates; {text[position:position + 8]!r} "
                    "is not a number"
                )
            coordinates.append(float(match.group()))
            position = match.end()
            if position < length:
                position += 1
        x, y = coordinates
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise CurveError(
                f"{option} key point ({x:g};{y:g}) is outside the [0;1] range, "
                "which upstream fails to configure"
            )
        if points and _table_index(points[-1][0], scale) >= _table_index(x, scale):
            raise CurveError(
                f"{option} key points ({points[-1][0]:g};{points[-1][1]:g}) and "
                f"({x:g};{y:g}) are too close together or not strictly "
                "increasing on the x axis, which upstream fails to configure"
            )
        points.append((x, y))
    return tuple(points)


def _identity_table(size: int) -> list[int]:
    return list(range(size))


def interpolate_natural(
    points: tuple[Point, ...], arithmetic: Arithmetic, size: int = LUT_SIZE
) -> list[int]:
    """``interpolate``: a natural cubic spline through the key points."""

    scale = size - 1
    count = len(points)
    if count == 0:
        return _identity_table(size)
    if count == 1:
        return [_clip(points[0][1] * scale, scale)] * size

    widths = [points[i + 1][0] - points[i][0] for i in range(count - 1)]

    # matrix is calloc'd upstream, so every entry not written below is zero;
    # that is what makes the first and last rows the natural end conditions.
    below = [0.0] * count
    main = [0.0] * count
    above = [0.0] * count
    right = [0.0] * count

    for i in range(1, count - 1):
        previous, current, following = (
            points[i - 1][1],
            points[i][1],
            points[i + 1][1],
        )
        right[i] = 6 * (
            (following - current) / widths[i] - (current - previous) / widths[i - 1]
        )

    main[0] = main[count - 1] = 1.0
    for i in range(1, count - 1):
        below[i] = widths[i - 1]
        main[i] = 2 * (widths[i - 1] + widths[i])
        above[i] = widths[i]

    for i in range(1, count):
        denominator = arithmetic.subtract_product(main[i], below[i], above[i - 1])
        factor = 1.0 / denominator if denominator else 1.0
        above[i] *= factor
        right[i] = arithmetic.subtract_product(right[i], below[i], right[i - 1]) * factor
    for i in range(count - 2, -1, -1):
        right[i] = arithmetic.subtract_product(right[i], above[i], right[i + 1])

    table = [0] * size
    first_x, first_y = points[0]
    for index in range(_table_index(first_x, scale)):
        table[index] = _clip(first_y * scale, scale)

    for i in range(count - 1):
        current_y = points[i][1]
        following_y = points[i + 1][1]
        a = current_y
        b = (
            (following_y - current_y) / widths[i]
            - widths[i] * right[i] / 2.0
            - widths[i] * (right[i + 1] - right[i]) / 6.0
        )
        c = right[i] / 2.0
        d = (right[i + 1] - right[i]) / (6.0 * widths[i])

        start = _table_index(points[i][0], scale)
        end = _table_index(points[i + 1][0], scale)
        for x in range(start, end + 1):
            # (x - x_start) * 1./scale, which * and / being left-associative
            # makes a division by scale rather than a multiply by its inverse.
            offset = (x - start) * 1.0 / scale
            # a + b*xx + c*xx*xx + d*xx*xx*xx, left to right, with each
            # multiply folded into the add that follows it.
            value = arithmetic.multiply_add(b, offset, a)
            value = arithmetic.multiply_add(c * offset, offset, value)
            value = arithmetic.multiply_add(d * offset * offset, offset, value)
            table[x] = _clip(value * scale, scale)

    last_x, last_y = points[-1]
    for index in range(_table_index(last_x, scale), size):
        table[index] = _clip(last_y * scale, scale)
    return table


def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def _pchip_edge(
    first_width: float,
    second_width: float,
    first_slope: float,
    second_slope: float,
    arithmetic: Arithmetic,
) -> float:
    """``pchip_edge_case``."""

    numerator = arithmetic.multiply_subtract(
        arithmetic.multiply_add(2, first_width, second_width),
        first_slope,
        first_width * second_slope,
    )
    derivative = numerator / (first_width + second_width)
    if _sign(derivative) != _sign(first_slope):
        return 0.0
    if _sign(first_slope) != _sign(second_slope) and abs(derivative) > 3.0 * abs(
        first_slope
    ):
        return 3.0 * first_slope
    return derivative


def _pchip_derivatives(
    widths: list[float], slopes: list[float], arithmetic: Arithmetic
) -> list[float]:
    """``pchip_find_derivatives`` over ``n = len(slopes)`` intervals."""

    count = len(slopes)
    derivatives = [0.0] * (count + 1)
    signs = [_sign(slope) for slope in slopes]
    for i in range(count - 1):
        if signs[i + 1] != signs[i] or slopes[i + 1] == 0 or slopes[i] == 0:
            derivatives[i + 1] = 0.0
        else:
            first = arithmetic.multiply_add(2, widths[i + 1], widths[i])
            second = arithmetic.multiply_add(2, widths[i], widths[i + 1])
            derivatives[i + 1] = (first + second) / (
                first / slopes[i] + second / slopes[i + 1]
            )
    derivatives[0] = _pchip_edge(
        widths[0], widths[1], slopes[0], slopes[1], arithmetic
    )
    derivatives[count] = _pchip_edge(
        widths[count - 1], widths[count - 2], slopes[count - 1], slopes[count - 2],
        arithmetic,
    )
    return derivatives


def _hermite_half(
    position: float, value: float, derivative: float, arithmetic: Arithmetic
) -> float:
    """``interp_cubic_hermite_half``."""

    squared = position * position
    cubed = squared * position
    return arithmetic.multiply_add(
        value,
        arithmetic.multiply_subtract(3.0, squared, 2.0 * cubed),
        derivative * (cubed - squared),
    )


def interpolate_pchip(
    points: tuple[Point, ...], arithmetic: Arithmetic, size: int = LUT_SIZE
) -> list[int]:
    """``interpolate_pchip``: monotonic piecewise cubic through the key points."""

    scale = size - 1
    count = len(points)
    if count == 0:
        return _identity_table(size)
    if count == 1:
        return [_clip(points[0][1] * scale, scale)] * size

    inputs = [point[0] * scale for point in points]
    values = [point[1] * scale for point in points]
    widths: list[float] = []
    slopes: list[float] = []
    for i in range(count - 1):
        width = inputs[i + 1] - inputs[i]
        widths.append(width)
        slopes.append((values[i + 1] - values[i]) / width)

    table = [0] * size
    if count == 2:
        slope = slopes[0]
        intercept = arithmetic.subtract_product(values[0], inputs[0], slope)
        for index in range(size):
            table[index] = _clip(
                arithmetic.multiply_add(index, slope, intercept), scale
            )
        return table

    derivatives = _pchip_derivatives(widths, slopes, arithmetic)

    x = 0
    if inputs[0] > 0:
        entry = _clip(values[0], scale)
        while x < inputs[0]:
            table[x] = entry
            x += 1

    for i in range(count - 1):
        start = inputs[i]
        end = inputs[i + 1]
        width = widths[i]
        while x < end:
            offset = (x - start) / width
            value = _hermite_half(
                1 - offset, values[i], -width * derivatives[i], arithmetic
            ) + _hermite_half(
                offset, values[i + 1], width * derivatives[i + 1], arithmetic
            )
            table[x] = _clip(value, scale)
            x += 1

    if x and x < size:
        entry = _clip(values[count - 1], scale)
        while x < size:
            table[x] = entry
            x += 1
    return table


INTERPOLATORS = {
    "natural": interpolate_natural,
    "pchip": interpolate_pchip,
}


@functools.lru_cache(maxsize=256)
def _interpolate_both(
    points: tuple[Point, ...], interpolation: str, size: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build one curve under both arithmetics, memoized on its inputs.

    The same key points appear once per layout a chain is compiled for, and a
    16-bit table is 65536 entries built twice, so this is worth remembering.
    """

    interpolator = INTERPOLATORS[interpolation]
    return tuple(
        tuple(interpolator(points, arithmetic, size)) for arithmetic in ARITHMETICS
    )  # type: ignore[return-value]


def build_table(
    points: tuple[Point, ...],
    interpolation: str,
    option: str,
    resolve_fusion: Callable[[], bool],
    size: int = LUT_SIZE,
) -> tuple[int, ...]:
    """Build one component's table under whichever arithmetic the host uses.

    Both evaluations are built and compared.  They almost always agree, and
    then the table is the same on every host and ``resolve_fusion`` is never
    called -- which is what keeps a curve's plan hash portable.

    They disagree when the curve passes exactly through a table-entry boundary,
    which ordinary key points do more often than the tiny ``double`` rounding
    gap suggests: ``0/0.05 1/1`` is the straight line ``y = (0.05 + 0.95x)``,
    and at input 155 that is exactly ``160/255``.  One evaluation order lands
    just below and the other just above, and the truncating ``CLIP`` turns that
    into 159 or 160.  Upstream's own output is not portable there, so the only
    way to be bit-exact against the pinned oracle is to ask which host this is.
    A deeper table has more entries to land on, so this happens more often
    rather than less.
    """

    fused, separate = _interpolate_both(points, interpolation, size)
    if fused == separate:
        return tuple(fused)
    entry = next(index for index in range(size) if fused[index] != separate[index])
    try:
        fuses = resolve_fusion()
    except CurveError:
        raise
    except Exception as error:  # pragma: no cover - depends on the host table
        raise CurveError(
            f"{option} produces {fused[entry]} or {separate[entry]} for input "
            f"{entry} depending on whether the host fuses a multiply-add, and "
            f"that could not be determined: {error}"
        ) from error
    return tuple(fused if fuses else separate)


def compose(component: tuple[int, ...], master: tuple[int, ...]) -> tuple[int, ...]:
    """Apply the master curve on top of one component's, as ``config_input`` does."""

    return tuple(master[value] for value in component)
