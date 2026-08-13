"""``vf_curves.c``'s table construction, in the precision upstream uses.

``curves`` is the one filter of this group that needs no IR extension.  Its
slice functions are ``dst[x + r] = graph[R][src[x + r]]`` and nothing else: one
256-entry table per colour channel, alpha copied.  Everything interesting
happens once, in ``config_input``, where the key points become a curve.  So the
lowering is a ``lut8`` like ``lutrgb``'s, and this module is the part that has
to be exact.

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
import math
import re


class CurveError(ValueError):
    """A curve specification is outside the accepted subset, or is refused."""


#: The 8-bit table upstream builds: 256 entries indexed by the input byte.
LUT_SIZE = 256
#: ``lut_size - 1``: what upstream calls ``scale``, the white value.
SCALE = LUT_SIZE - 1

Point = tuple[float, float]


def _fused(left: float, right: float, addend: float) -> float:
    """One correctly rounded ``double`` fused multiply-add.

    ``math.fma`` only exists from Python 3.13, and this has to run on 3.12, so
    the product and sum are formed exactly as rationals and rounded once.
    ``int.__truediv__`` is correctly rounded, which is what makes the final
    conversion the single rounding a hardware ``fma`` performs.
    """

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


def _clip(value: float) -> int:
    """``CLIP(v)`` at 8-bit depth: ``av_clip_uint8`` of a truncating conversion."""

    if not value > 0.0:
        return 0
    if value >= 255.0:
        return 255
    return int(value)


def _table_index(value: float) -> int:
    """``(int)(point->x * scale)``, which truncates toward zero."""

    return int(value * SCALE)


#: A plain decimal literal.  ``av_strtod`` additionally accepts SI postfixes,
#: so ``0.5k`` means 500 upstream; those are refused rather than honoured, for
#: the same reason ``lutyuv=r=`` is.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def parse_points(text: str, option: str) -> tuple[Point, ...]:
    """Reproduce ``parse_points_str``, refusing what it would misread.

    Upstream reads a number, steps over exactly one character, reads another,
    and steps again -- so the separators are positional rather than meaningful
    and ``0/0 1/1`` and ``0-0*1-1`` parse the same.  That control flow is kept;
    what is not kept is ``av_strtod`` accepting anything but a decimal literal.
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
        if points and _table_index(points[-1][0]) >= _table_index(x):
            raise CurveError(
                f"{option} key points ({points[-1][0]:g};{points[-1][1]:g}) and "
                f"({x:g};{y:g}) are too close together or not strictly "
                "increasing on the x axis, which upstream fails to configure"
            )
        points.append((x, y))
    return tuple(points)


def _identity_table() -> list[int]:
    return list(range(LUT_SIZE))


def interpolate_natural(points: tuple[Point, ...], arithmetic: Arithmetic) -> list[int]:
    """``interpolate``: a natural cubic spline through the key points."""

    count = len(points)
    if count == 0:
        return _identity_table()
    if count == 1:
        return [_clip(points[0][1] * SCALE)] * LUT_SIZE

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

    table = [0] * LUT_SIZE
    first_x, first_y = points[0]
    for index in range(_table_index(first_x)):
        table[index] = _clip(first_y * SCALE)

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

        start = _table_index(points[i][0])
        end = _table_index(points[i + 1][0])
        for x in range(start, end + 1):
            # (x - x_start) * 1./scale, which * and / being left-associative
            # makes a division by scale rather than a multiply by its inverse.
            offset = (x - start) * 1.0 / SCALE
            # a + b*xx + c*xx*xx + d*xx*xx*xx, left to right, with each
            # multiply folded into the add that follows it.
            value = arithmetic.multiply_add(b, offset, a)
            value = arithmetic.multiply_add(c * offset, offset, value)
            value = arithmetic.multiply_add(d * offset * offset, offset, value)
            table[x] = _clip(value * SCALE)

    last_x, last_y = points[-1]
    for index in range(_table_index(last_x), LUT_SIZE):
        table[index] = _clip(last_y * SCALE)
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


def interpolate_pchip(points: tuple[Point, ...], arithmetic: Arithmetic) -> list[int]:
    """``interpolate_pchip``: monotonic piecewise cubic through the key points."""

    count = len(points)
    if count == 0:
        return _identity_table()
    if count == 1:
        return [_clip(points[0][1] * SCALE)] * LUT_SIZE

    inputs = [point[0] * SCALE for point in points]
    values = [point[1] * SCALE for point in points]
    widths: list[float] = []
    slopes: list[float] = []
    for i in range(count - 1):
        width = inputs[i + 1] - inputs[i]
        widths.append(width)
        slopes.append((values[i + 1] - values[i]) / width)

    table = [0] * LUT_SIZE
    if count == 2:
        slope = slopes[0]
        intercept = arithmetic.subtract_product(values[0], inputs[0], slope)
        for index in range(LUT_SIZE):
            table[index] = _clip(arithmetic.multiply_add(index, slope, intercept))
        return table

    derivatives = _pchip_derivatives(widths, slopes, arithmetic)

    x = 0
    if inputs[0] > 0:
        entry = _clip(values[0])
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
            table[x] = _clip(value)
            x += 1

    if x and x < LUT_SIZE:
        entry = _clip(values[count - 1])
        while x < LUT_SIZE:
            table[x] = entry
            x += 1
    return table


INTERPOLATORS = {
    "natural": interpolate_natural,
    "pchip": interpolate_pchip,
}


def build_table(
    points: tuple[Point, ...],
    interpolation: str,
    option: str,
    resolve_fusion: Callable[[], bool],
) -> tuple[int, ...]:
    """Build one component's table under whichever arithmetic the host uses.

    Both evaluations are built and compared.  They almost always agree, and
    then the table is the same on every host and ``resolve_fusion`` is never
    called -- which is what keeps a curve's plan hash portable.

    They disagree when the curve passes exactly through a byte boundary, which
    ordinary key points do more often than the tiny ``double`` rounding gap
    suggests: ``0/0.05 1/1`` is the straight line ``y = (0.05 + 0.95x)``, and at
    input 155 that is exactly ``160/255``.  One evaluation order lands just
    below and the other just above, and the truncating ``CLIP`` turns that into
    159 or 160.  Upstream's own output is not portable there, so the only way
    to be bit-exact against the pinned oracle is to ask which host this is.
    """

    interpolator = INTERPOLATORS[interpolation]
    fused, separate = (interpolator(points, arithmetic) for arithmetic in ARITHMETICS)
    if fused == separate:
        return tuple(fused)
    entry = next(index for index in range(LUT_SIZE) if fused[index] != separate[index])
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
