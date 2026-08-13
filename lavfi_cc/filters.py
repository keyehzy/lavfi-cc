"""Per-filter eligibility checks and lowering into pixel-IR operations.

Each entry in :data:`LOWERERS` turns one parsed filter invocation into the
operations it contributes, or raises :class:`LoweringError` explaining exactly
why that invocation is outside the accepted subset.  Keeping this layer free of
graph-level concerns lets both the fusion frontend and the analysis-only
scanner ask the same question about a filter.

A lowerer also receives the layout the filter will run in, because an option's
meaning can depend on it: ``negate=components=y`` is valid in ``yuv420p`` and a
configuration failure in ``rgba``, and the reverse holds for
``components=r``.  The layout is ``None`` when the working format is still
undecided, which happens only on the scanner's reporting path; a lowerer then
checks what it can decide without a format and leaves the rest to the caller,
which has already refused to fuse a run whose format it cannot name.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
from typing import Any, Callable

from .curves import CurveError
from .curves import build_table as build_curve_table
from .curves import compose as compose_curves
from .curves import parse_points as parse_curve_points
from .expr import ExprBuilder, ExprProgram, multiply_is_exact
from .expressions import ExpressionError, build_lut
from .ir import CHANNELS, Operation, source_ref
from .layouts import PixelLayout
from .parser import FilterInvocation
from .target import UnknownTargetError, target_fuses_multiply_add


class LoweringError(ValueError):
    def __init__(self, code: str, message: str, option: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.option = option


@dataclass(frozen=True)
class Lowered:
    operations: tuple[Operation, ...]
    canonical: str
    removes_color_side_data: bool = False


def format_value(invocation: FilterInvocation) -> str:
    """Return the single literal pixel format named by a ``format`` filter."""

    named = invocation.named_options()
    positional = invocation.positional_options()
    if named and positional:
        raise LoweringError(
            "invalid_format_boundary", "format boundary mixes named and positional options"
        )
    if positional:
        if len(positional) != 1:
            raise LoweringError(
                "invalid_format_boundary", "format boundary must name exactly one pixel format"
            )
        value = positional[0].value
    else:
        if set(named) != {"pix_fmts"}:
            raise LoweringError(
                "invalid_format_boundary",
                "format boundary must use format=<pixel-format> or format=pix_fmts=<pixel-format>",
            )
        value = named["pix_fmts"].value
    if not value or not all(character.isalnum() or character == "_" for character in value):
        raise LoweringError(
            "invalid_format_boundary",
            "format boundary must select one literal pixel format (lists are unsupported)",
        )
    return value.lower()


def _reject_positionals(invocation: FilterInvocation) -> dict[str, Any]:
    if invocation.positional_options():
        raise LoweringError(
            "positional_options",
            f"{invocation.name} requires explicit key=value options in the accepted subset",
        )
    return invocation.named_options()


def _check_options(invocation: FilterInvocation, allowed: set[str]) -> dict[str, Any]:
    if invocation.option_error is not None:
        raise LoweringError("unparsed_options", invocation.option_error)
    options = _reject_positionals(invocation)
    unknown = sorted(set(options) - allowed)
    if unknown:
        option = unknown[0]
        if option == "enable":
            message = "timeline expressions are not static and cannot be fused"
            code = "runtime_option"
        else:
            message = f"option {option!r} is not supported for {invocation.name}"
            code = "unsupported_option"
        raise LoweringError(code, message, option)
    return options


def _parse_bool(value: str, option: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LoweringError("invalid_value", f"expected a boolean, got {value!r}", option)


def _parse_number(value: str, option: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise LoweringError("invalid_value", f"expected a number, got {value!r}", option) from error
    if not math.isfinite(number):
        raise LoweringError("non_finite", "non-finite numeric values are not supported", option)
    if not minimum <= number <= maximum:
        raise LoweringError(
            "out_of_range", f"value {value!r} is outside [{minimum:g}, {maximum:g}]", option
        )
    return 0.0 if number == 0.0 else number


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _levels_rounding_is_target_independent(
    input_min: int, output_min: int, coefficient: float
) -> bool:
    """Return whether contracted and separate float32 evaluation agree as bytes."""

    def quantize(value: float) -> int:
        return max(0, min(255, int(value)))

    for pixel in range(256):
        product = (pixel - input_min) * coefficient
        contracted = _float32(product + output_min)
        separate = _float32(_float32(product) + output_min)
        if quantize(contracted) != quantize(separate):
            return False
    return True


#: A plain decimal literal, which is the whole static subset of an option that
#: upstream evaluates with ``av_expr_eval``.  Anything richer either names a
#: per-frame variable or reaches an evaluator this compiler does not reproduce.
_NUMERIC_LITERAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _parse_static_expression(value: str, option: str) -> float:
    """Parse an option whose value upstream treats as an expression.

    Only a literal is accepted.  ``eq`` and ``hue`` evaluate these with the
    frame counter, timestamp, and frame rate in scope, so anything referring to
    a variable is a different value on every frame and cannot be baked into a
    kernel at all.
    """

    if not _NUMERIC_LITERAL.fullmatch(value.strip()):
        raise LoweringError(
            "dynamic_expression",
            f"{option} must be a plain number in the accepted subset; "
            f"{value!r} is an expression that upstream may re-evaluate per frame",
            option,
        )
    number = float(value)
    if not math.isfinite(number):
        raise LoweringError(
            "non_finite", "non-finite numeric values are not supported", option
        )
    return number


def _clipf(value: float, minimum: float, maximum: float) -> float:
    """Reproduce ``av_clipf``, which clamps in binary32 rather than binary64.

    Its parameters are ``float``, so a ``double`` expression result is rounded
    to binary32 on the way in and the bounds are the binary32 neighbours of the
    literals in the source.  Every ``eq`` option passes through this, so the
    rounding is part of the filter's observable behaviour.
    """

    return _float32(
        min(max(_float32(value), _float32(minimum)), _float32(maximum))
    )


def _identity_table() -> tuple[int, ...]:
    return tuple(range(256))


def _table_hash(table: tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(table)).hexdigest()


#: Every component name the accepted filters use, and the logical channel it
#: selects.  The two families never overlap and both order their channels the
#: same way, so one table serves ``r+g+b+a`` and ``y+u+v+a`` alike.
COMPONENT_CHANNELS = {"r": 0, "y": 0, "g": 1, "u": 1, "b": 2, "v": 2, "a": 3}

#: What a caller may spell when the working format is still undecided.
_COMPONENT_FAMILIES = (frozenset("rgba"), frozenset("yuva"))


def _negate_channels(
    invocation: FilterInvocation, layout: PixelLayout | None
) -> set[int]:
    """Return the logical channels ``negate`` inverts in this layout.

    Upstream validates an explicit mask against the format and fails the graph
    outright when a component is missing; the default mask names every colour
    component of whichever family the format belongs to and skips that check.
    """

    options = invocation.named_options()
    raw = options.get("components")
    if raw is None:
        return {0, 1, 2}

    tokens = raw.value.split("+")
    if not tokens or any(token not in COMPONENT_CHANNELS for token in tokens):
        raise LoweringError(
            "invalid_components",
            "components must be a + separated combination of r, g, b, and a "
            "for an RGB format, or y, u, v, and a for a YUV one",
            "components",
        )
    if len(set(tokens)) != len(tokens):
        raise LoweringError(
            "invalid_components", "components must not contain duplicates", "components"
        )

    if layout is None:
        # Without a format only the vocabulary is decidable: no format offers
        # both families, but which one it offers is not yet known.
        if not any(set(tokens) <= family for family in _COMPONENT_FAMILIES):
            raise LoweringError(
                "invalid_components",
                "components must not mix the RGB and YUV component names",
                "components",
            )
        return {COMPONENT_CHANNELS[token] for token in tokens}

    available = layout.components
    unavailable = [token for token in tokens if token not in available]
    if unavailable:
        raise LoweringError(
            "component_not_available",
            f"negate cannot select {unavailable[0]!r} in {layout.name!r}, which "
            f"carries {'+'.join(available)}; upstream fails to configure this graph",
            "components",
        )
    return {available[token] for token in tokens}


def _check_negate_alpha(
    invocation: FilterInvocation, layout: PixelLayout | None
) -> None:
    """Reject the legacy plane option only where it really negates alpha.

    ``negate_alpha`` sets a *plane* mask.  Packed RGB ignores it in favour of
    its component mask, and a layout without an alpha plane has nothing for it
    to select, so in both cases it is inert.  On ``gbrap`` it does negate alpha,
    which is a different meaning for the same option.  An explicit ``components``
    replaces the plane mask outright, so the option cannot matter there either.
    """

    options = invocation.named_options()
    negate_alpha = options.get("negate_alpha")
    if negate_alpha is None:
        return
    enabled = _parse_bool(negate_alpha.value, "negate_alpha")
    if (
        enabled
        and "components" not in options
        and layout is not None
        and layout.planar
        and layout.has_alpha
    ):
        raise LoweringError(
            "planar_negate_alpha",
            f"negate_alpha sets a plane mask, so it negates alpha in "
            f"{layout.name!r} unlike in packed RGB; spell the intent as "
            "components=r+g+b+a instead",
            "negate_alpha",
        )


def _lower_negate(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    _check_options(invocation, {"components", "negate_alpha"})
    channels = _negate_channels(invocation, layout)
    _check_negate_alpha(invocation, layout)

    identity = _identity_table()
    inverse = tuple(255 - value for value in range(256))
    tables = tuple(
        inverse if channel in channels else identity for channel in range(4)
    )
    operation = Operation("lut8", {"tables": tables}, source_ref(index, invocation))
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "lookup_u8"},
        source_ref(index, invocation),
    )
    names = layout.component_names if layout is not None else CHANNELS
    ordered = "+".join(
        names[channel] or CHANNELS[channel] for channel in sorted(channels)
    )
    return Lowered((operation, quantize), f"negate=components={ordered}")


#: Component names and per-channel ``[minval, maxval]`` range of each
#: ``vf_lut.c`` entry point at 8-bit depth.
#:
#: The two entry points are the same filter with a different component
#: vocabulary and a different range.  ``lutrgb`` falls into ``config_props``'s
#: default branch, which gives every component ``0..255``; ``lutyuv`` falls into
#: the YUV branch, which gives luma ``16..235`` and each chroma channel
#: ``16..240``.  Alpha is full range in both, and the final ``av_clip((int)res,
#: 0, max[A])`` is ``[0, 255]`` either way.
#:
#: Upstream also accepts ``c0``..``c3`` as slot-numbered aliases for these
#: names, and the RGB names alias the YUV slots in ``lutyuv`` and vice versa.
#: Both spellings are refused here: they name the same option twice with no
#: added expressiveness, and ``lutyuv=r=…`` silently means luma.
_LUT_VARIANTS: dict[str, tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = {
    "lutrgb": (CHANNELS, ((0, 255), (0, 255), (0, 255), (0, 255))),
    "lutyuv": (("y", "u", "v", "a"), ((16, 235), (16, 240), (16, 240), (0, 255))),
}


def _lower_lut(invocation: FilterInvocation, index: int) -> Lowered:
    """Lower one ``vf_lut.c`` entry point into a four-channel byte table."""

    names, ranges = _LUT_VARIANTS[invocation.name]
    options = _check_options(invocation, set(names))
    expressions = {name: "clipval" for name in names}
    expressions.update({name: option.value for name, option in options.items()})
    tables: list[tuple[int, ...]] = []
    for name, value_range in zip(names, ranges, strict=True):
        try:
            tables.append(build_lut(expressions[name], value_range))
        except ExpressionError as error:
            raise LoweringError("unsupported_expression", str(error), name) from error
    operation = Operation("lut8", {"tables": tuple(tables)}, source_ref(index, invocation))
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "truncate_toward_zero_then_saturate"},
        source_ref(index, invocation),
    )
    canonical = f"{invocation.name}=" + ":".join(
        f"{name}=table:{_table_hash(table)}"
        for name, table in zip(names, tables, strict=True)
    )
    # vf_lut.c drops colour-dependent side data unconditionally, for every one
    # of its entry points.
    return Lowered((operation, quantize), canonical, removes_color_side_data=True)


def _lower_lutrgb(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    return _lower_lut(invocation, index)


def _lower_lutyuv(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    return _lower_lut(invocation, index)


#: ``eq``'s options, their defaults, and the range ``av_clipf`` clamps them to.
#:
#: Out-of-range values are clamped rather than refused, because that is what
#: upstream does; refusing them would reject graphs FFmpeg accepts.
_EQ_OPTIONS: dict[str, tuple[str, float, float]] = {
    "contrast": ("1.0", -1000.0, 1000.0),
    "brightness": ("0.0", -1.0, 1.0),
    "saturation": ("1.0", 0.0, 3.0),
    "gamma": ("1.0", 0.1, 10.0),
    "gamma_r": ("1.0", 0.1, 10.0),
    "gamma_g": ("1.0", 0.1, 10.0),
    "gamma_b": ("1.0", 0.1, 10.0),
    "gamma_weight": ("1.0", 0.0, 1.0),
}

#: How many binary32 steps of slack the gamma table must tolerate.
#:
#: ``create_lut`` is the only accepted path that calls ``pow``, and a libm's
#: ``pow`` is not bit-exact across implementations.  A table is only emitted
#: when perturbing every ``pow`` result by this many ULPs leaves all 256 bytes
#: unchanged, so a kernel cannot disagree with the oracle over a rounding this
#: compiler does not control.
_GAMMA_ULP_SLACK = 4


def _next_after(value: float, steps: int) -> float:
    """Return the binary64 *steps* ULPs away from *value*."""

    for _ in range(abs(steps)):
        value = math.nextafter(value, math.inf if steps > 0 else -math.inf)
    return value


def _eq_process_table(contrast: float, brightness: float) -> tuple[int, ...]:
    """Reproduce ``process_c``, which is integer arithmetic end to end."""

    def c_divide(numerator: int, denominator: int) -> int:
        # C integer division truncates toward zero; Python's floors.
        quotient = abs(numerator) // abs(denominator)
        return -quotient if (numerator < 0) != (denominator < 0) else quotient

    scaled = int(contrast * 256 * 16)
    offset = (
        c_divide(int(100.0 * brightness + 100.0) * 511, 200)
        - 128
        - c_divide(scaled, 32)
    )
    table: list[int] = []
    for value in range(256):
        # `pel & ~255` then `(-pel) >> 31` is a saturating cast to uint8.
        pixel = ((value * scaled) >> 12) + offset
        table.append(max(0, min(255, pixel)))
    return tuple(table)


def _eq_gamma_table(
    contrast: float, brightness: float, gamma: float, gamma_weight: float, option: str
) -> tuple[int, ...]:
    """Reproduce ``create_lut``, and refuse it when ``pow`` decides a byte."""

    exponent = 1.0 / gamma
    linear_weight = 1.0 - gamma_weight

    def quantize(base: float, powed: float) -> int:
        value = base * linear_weight + powed * gamma_weight
        return 255 if value >= 1.0 else int(256.0 * value)

    table: list[int] = []
    for value in range(256):
        base = contrast * (value / 255.0 - 0.5) + 0.5 + brightness
        if base <= 0.0:
            table.append(0)
            continue
        powed = math.pow(base, exponent)
        byte = quantize(base, powed)
        for steps in (-_GAMMA_ULP_SLACK, _GAMMA_ULP_SLACK):
            if quantize(base, _next_after(powed, steps)) != byte:
                raise LoweringError(
                    "libm_sensitive_gamma",
                    f"the gamma table entry for {value} changes when pow() moves "
                    f"by {_GAMMA_ULP_SLACK} ULPs, so it depends on a libm rounding "
                    "this compiler cannot guarantee matches the oracle",
                    option,
                )
        table.append(byte)
    return tuple(table)


def _eq_channel_table(
    contrast: float,
    brightness: float,
    gamma: float,
    gamma_weight: float,
    option: str,
) -> tuple[tuple[int, ...], str]:
    """Pick the same one of ``check_values``' three paths that upstream picks.

    The three are genuinely different mappings, not refinements of each other:
    the all-default case copies the plane, while ``process_c`` with those same
    values would subtract one from every sample.
    """

    if contrast == 1.0 and brightness == 0.0 and gamma == 1.0:
        return _identity_table(), "copy"
    if gamma == 1.0 and abs(contrast) < 7.9:
        return _eq_process_table(contrast, brightness), "process"
    return (
        _eq_gamma_table(contrast, brightness, gamma, gamma_weight, option),
        "lut",
    )


def _lower_eq(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    options = _check_options(invocation, set(_EQ_OPTIONS) | {"eval"})

    evaluation = options.get("eval")
    if evaluation is not None and evaluation.value.lower() != "init":
        raise LoweringError(
            "runtime_option",
            "eval=frame re-evaluates the options on every frame, so they are "
            "not static and cannot be baked into a kernel",
            "eval",
        )

    values: dict[str, float] = {}
    for name, (default, minimum, maximum) in _EQ_OPTIONS.items():
        raw = options[name].value if name in options else default
        values[name] = _clipf(_parse_static_expression(raw, name), minimum, maximum)

    # param[0] is luma; param[1] and param[2] are Cb and Cr, whose contrast is
    # the saturation and whose brightness upstream leaves at zero. Alpha is
    # copied, so its table is the identity.
    channels = (
        (values["contrast"], values["brightness"],
         values["gamma"] * values["gamma_g"], "gamma"),
        (values["saturation"], 0.0,
         math.sqrt(values["gamma_b"] / values["gamma_g"]), "gamma_b"),
        (values["saturation"], 0.0,
         math.sqrt(values["gamma_r"] / values["gamma_g"]), "gamma_r"),
    )
    tables: list[tuple[int, ...]] = []
    paths: list[str] = []
    for contrast, brightness, gamma, option in channels:
        table, path = _eq_channel_table(
            contrast, brightness, gamma, values["gamma_weight"], option
        )
        tables.append(table)
        paths.append(path)
    tables.append(_identity_table())

    operation = Operation(
        "lut8", {"tables": tuple(tables)}, source_ref(index, invocation)
    )
    quantize = Operation(
        "quantize_rgba8", {"mode": "lookup_u8"}, source_ref(index, invocation)
    )
    canonical = (
        "eq="
        + ":".join(f"{name}={values[name].hex()}" for name in _EQ_OPTIONS)
        + ":eval=init:paths="
        + "+".join(paths)
    )
    return Lowered((operation, quantize), canonical)


#: ``hue`` clamps saturation and brightness to this range before using them.
_HUE_LIMIT = 10.0

#: Binary32 slack the hue coefficients must tolerate, for the same reason
#: :data:`_GAMMA_ULP_SLACK` exists: ``sin`` and ``cos`` are libm calls whose
#: last bits this compiler does not control.  Here the result is rounded to an
#: integer, so the slack only bites when the scaled coefficient lands within a
#: few ULPs of a half-integer.
_HUE_ULP_SLACK = 4


def _lrint(value: float) -> int:
    """``lrint`` under the default rounding mode: nearest, ties to even."""

    return round(value)


def _hue_coefficient(
    trigonometric: float,
    saturation: float,
    option: str,
    name: str,
    exact: bool,
) -> int:
    """Scale one of ``sin``/``cos`` into 16.16, refusing a decisive rounding.

    ``exact`` says the trigonometric value is not an approximation, so no libm
    can disagree about it and the slack does not apply.  Without that the guard
    would refuse a large share of ordinary graphs: at a zero angle the product
    is a multiple of one sixteenth for any saturation, so it lands exactly on a
    rounding tie about one time in eight, and a tie moves under any
    perturbation however certain the value is.
    """

    coefficient = _lrint(trigonometric * (1 << 16) * saturation)
    if exact:
        return coefficient
    for steps in (-_HUE_ULP_SLACK, _HUE_ULP_SLACK):
        moved = _next_after(trigonometric, steps) * (1 << 16) * saturation
        if _lrint(moved) != coefficient:
            raise LoweringError(
                "libm_sensitive_hue",
                f"the {name} coefficient changes when {name}() moves by "
                f"{_HUE_ULP_SLACK} ULPs, so it depends on a libm rounding this "
                "compiler cannot guarantee matches the oracle",
                option,
            )
    return coefficient


def _lower_hue(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    options = _check_options(invocation, {"h", "H", "s", "b"})
    if "h" in options and "H" in options:
        raise LoweringError(
            "conflicting_options",
            "h and H are incompatible and upstream fails to configure a graph "
            "naming both",
            "H",
        )

    # av_clip and av_clipf both land on the range bound exactly, so a single
    # clamp describes either.
    saturation = min(
        _HUE_LIMIT,
        max(
            -_HUE_LIMIT,
            _float32(
                _parse_static_expression(
                    options["s"].value if "s" in options else "1", "s"
                )
            ),
        ),
    )
    brightness = min(
        _HUE_LIMIT,
        max(
            -_HUE_LIMIT,
            _float32(
                _parse_static_expression(
                    options["b"].value if "b" in options else "0", "b"
                )
            ),
        ),
    )

    # h is degrees and H is radians. hue_deg is a float, so the conversion to
    # radians rounds to binary32 twice: once on the option, once on the result.
    if "h" in options:
        degrees = _float32(_parse_static_expression(options["h"].value, "h"))
        angle = _float32(degrees * math.pi / 180)
        option = "h"
    elif "H" in options:
        angle = _float32(_parse_static_expression(options["H"].value, "H"))
        option = "H"
    else:
        angle = 0.0
        option = "s"

    # C requires cos(0) to be exactly 1 and sin(0) exactly 0, so at a zero
    # angle -- which is what hue=s=… and hue=b=… mean -- the coefficients carry
    # no libm uncertainty at all.
    exact = angle == 0.0
    cosine = _hue_coefficient(math.cos(angle), saturation, option, "cos", exact)
    sine = _hue_coefficient(math.sin(angle), saturation, option, "sin", exact)

    # Luma only carries the brightness offset. Upstream copies the plane when
    # brightness is zero and applies this table otherwise; the table is the
    # identity at zero, so one path describes both.
    luma = tuple(
        max(0, min(255, int(value + brightness * 25.5))) for value in range(256)
    )
    identity = _identity_table()
    lut = Operation(
        "lut8",
        {"tables": (luma, identity, identity, identity)},
        source_ref(index, invocation),
    )
    lut_quantize = Operation(
        "quantize_rgba8", {"mode": "lookup_u8"}, source_ref(index, invocation)
    )
    rotate = Operation(
        "chroma_rotate_i32",
        {"cosine": cosine, "sine": sine},
        source_ref(index, invocation),
    )
    rotate_quantize = Operation(
        "quantize_rgba8",
        {"mode": "saturate_i32_to_u8"},
        source_ref(index, invocation),
    )
    canonical = f"hue=cos={cosine}:sin={sine}:b={brightness.hex()}"
    return Lowered((lut, lut_quantize, rotate, rotate_quantize), canonical)


def _lower_colorlevels(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    point_options = {
        f"{channel}{suffix}"
        for channel in CHANNELS
        for suffix in ("imin", "imax", "omin", "omax")
    }
    options = _check_options(invocation, point_options | {"preserve"})
    preserve = options.get("preserve")
    if preserve is not None and preserve.value.lower() not in {"none", "0"}:
        raise LoweringError(
            "unsupported_preserve", "only preserve=none is pixel-local in the MVP", "preserve"
        )

    input_min: list[int] = []
    input_max: list[int] = []
    output_min: list[int] = []
    output_max: list[int] = []
    coefficients: list[str] = []
    for channel in CHANNELS:
        defaults = {"imin": 0.0, "imax": 1.0, "omin": 0.0, "omax": 1.0}
        parsed: dict[str, float] = {}
        for suffix, default in defaults.items():
            name = f"{channel}{suffix}"
            if name in options:
                lower = -1.0 if suffix in {"imin", "imax"} else 0.0
                parsed[suffix] = _parse_number(options[name].value, name, lower, 1.0)
            else:
                parsed[suffix] = default
        if parsed["imin"] < 0 or parsed["imax"] < 0:
            offending = f"{channel}{'imin' if parsed['imin'] < 0 else 'imax'}"
            raise LoweringError(
                "frame_global_extrema",
                "negative colorlevels input points trigger a per-frame extrema scan",
                offending,
            )
        imin = round(parsed["imin"] * 255.0)
        imax = round(parsed["imax"] * 255.0)
        omin = round(parsed["omin"] * 255.0)
        omax = round(parsed["omax"] * 255.0)
        if imax == imin:
            raise LoweringError(
                "degenerate_levels",
                f"{channel} input endpoints both quantize to {imin}",
                f"{channel}imax",
            )
        coefficient = _float32((omax - omin) / float(imax - imin))
        if not _levels_rounding_is_target_independent(imin, omin, coefficient):
            raise LoweringError(
                "target_sensitive_levels",
                f"{channel} channel produces different bytes with contracted and "
                "separate binary32 multiply-add evaluation",
                f"{channel}imin",
            )
        input_min.append(imin)
        input_max.append(imax)
        output_min.append(omin)
        output_max.append(omax)
        coefficients.append(coefficient.hex())

    matrix = [
        [coefficients[row] if row == column else "0x0.0p+0" for column in range(4)]
        for row in range(4)
    ]
    offsets = [
        {"input": input_min[channel], "output": output_min[channel]}
        for channel in range(4)
    ]
    operation = Operation(
        "matrix4x4",
        {
            "evaluation": "levels_f32_fma",
            "coefficients": matrix,
            "offsets": offsets,
            "input_max": input_max,
            "output_max": output_max,
        },
        source_ref(index, invocation),
    )
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "truncate_toward_zero_then_saturate"},
        source_ref(index, invocation),
    )
    canonical = (
        "colorlevels="
        + ":".join(
            f"{channel}imin={input_min[position]}:{channel}imax={input_max[position]}:"
            f"{channel}omin={output_min[position]}:{channel}omax={output_max[position]}"
            for position, channel in enumerate(CHANNELS)
        )
        + ":preserve=none"
    )
    return Lowered((operation, quantize), canonical)


def _lower_colorchannelmixer(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    coefficient_names = {output + input_ for output in CHANNELS for input_ in CHANNELS}
    options = _check_options(invocation, coefficient_names | {"pc", "pa"})
    pc = options.get("pc")
    if pc is not None and pc.value.lower() not in {"none", "0"}:
        raise LoweringError(
            "unsupported_preserve", "only pc=none is supported by the MVP", "pc"
        )
    if "pa" in options:
        _parse_number(options["pa"].value, "pa", 0.0, 1.0)

    coefficients: list[list[float]] = []
    coefficient_hex: list[list[str]] = []
    tables: list[list[tuple[int, ...]]] = []
    for output_position, output in enumerate(CHANNELS):
        row: list[float] = []
        row_hex: list[str] = []
        row_tables: list[tuple[int, ...]] = []
        for input_position, input_ in enumerate(CHANNELS):
            name = output + input_
            default = 1.0 if output_position == input_position else 0.0
            coefficient = (
                _parse_number(options[name].value, name, -2.0, 2.0)
                if name in options
                else default
            )
            row.append(coefficient)
            row_hex.append(coefficient.hex())
            row_tables.append(tuple(round(value * coefficient) for value in range(256)))
        coefficients.append(row)
        coefficient_hex.append(row_hex)
        tables.append(row_tables)

    operation = Operation(
        "matrix4x4",
        {
            "evaluation": "sum_i32_terms_rounded_ties_even",
            "coefficients": coefficient_hex,
            "offsets": [0, 0, 0, 0],
            "contribution_tables": tables,
        },
        source_ref(index, invocation),
    )
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "saturate_i32_to_u8"},
        source_ref(index, invocation),
    )
    canonical = "colorchannelmixer=" + ":".join(
        f"{output}{input_}={coefficient_hex[o][i]}"
        for o, output in enumerate(CHANNELS)
        for i, input_ in enumerate(CHANNELS)
    ) + ":pc=none"
    return Lowered((operation, quantize), canonical)


def _expr_operations(
    program: ExprProgram, invocation: FilterInvocation, index: int
) -> tuple[Operation, Operation]:
    """Wrap one expression program in the IR's transform/quantize pair.

    The quantization is per channel and already inside the program, because an
    expression's channels need not agree on one rule; the mode here only says
    so.
    """

    reference = source_ref(index, invocation)
    return (
        Operation("expr_f32", {"program": program.as_dict()}, reference),
        Operation("quantize_rgba8", {"mode": "expression_outputs"}, reference),
    )


#: ``colorbalance``'s nine range options, and which logical channel each drives.
#:
#: Upstream names them by the colour axis each one shifts -- ``rs`` is the red
#: end of the cyan/red range -- and passes them to ``get_component`` in the
#: order red, green, blue.
_COLORBALANCE_RANGES = (("r", 0), ("g", 1), ("b", 2))
_COLORBALANCE_WEIGHTS = (("s", "shadows"), ("m", "midtones"), ("h", "highlights"))

#: ``get_component``'s three constants: ``a``, ``b``, and ``scale``.
_BALANCE_SLOPE = 4.0
_BALANCE_PIVOT = 0.333
_BALANCE_SCALE = 0.7

#: Widest value reaching ``get_component``'s multiply by ``_BALANCE_SLOPE``.
#:
#: ``l`` is a sum of two channel values already divided by 255, so it lies in
#: ``[0, 2]``, and each of the four multiplicands is ``l`` offset by a constant
#: below one.  Two bounds every one of them.
_BALANCE_SLOPE_OPERAND_BOUND = 2.0


def _colorbalance_program(weights: dict[str, float]) -> ExprProgram:
    """Transcribe ``vf_colorbalance.c``'s ``get_component`` for all three channels.

    The four multiply-adds below are the only ones upstream contains, and each
    scales by ``4.f``.  Scaling by a power of two is exact, so contracting the
    multiply into the add rounds the same value once either way and this
    program means the same thing whether or not the host fuses -- which is why
    it takes no target argument.  :func:`multiply_is_exact` is asked rather
    than asserted in a comment.
    """

    if not multiply_is_exact(_BALANCE_SLOPE, _BALANCE_SLOPE_OPERAND_BOUND):
        raise LoweringError(
            "target_sensitive_balance",
            "colorbalance's multiply-adds are no longer exact, so whether the "
            "host fuses them would decide output bytes",
        )

    builder = ExprBuilder()
    zero = builder.const(0.0)
    one = builder.const(1.0)
    half = builder.const(0.5)
    slope = builder.const(_BALANCE_SLOPE)
    pivot = builder.const(_BALANCE_PIVOT)
    scale = builder.const(_BALANCE_SCALE)
    maximum = builder.const(255.0)

    def scaled(difference: int) -> int:
        """``av_clipf((difference) * a + 0.5f, 0.f, 1.f)``."""

        return builder.clipf(
            builder.add(builder.mul(difference, slope), half), zero, one
        )

    values = [builder.div(builder.channel(channel), maximum) for channel in range(3)]
    lightness = builder.add(builder.max3(*values), builder.min3(*values))

    # s *= av_clipf((b - l) * a + 0.5f, 0, 1) * scale
    shadow_factor = builder.mul(scaled(builder.sub(pivot, lightness)), scale)
    # m *= av_clipf((l - b) * a + .5, 0, 1) * av_clipf((1 - l - b) * a + .5, 0, 1) * scale
    midtone_factor = builder.mul(
        builder.mul(
            scaled(builder.sub(lightness, pivot)),
            scaled(builder.sub(builder.sub(one, lightness), pivot)),
        ),
        scale,
    )
    # h *= av_clipf((l + b - 1) * a + 0.5f, 0, 1) * scale
    highlight_factor = builder.mul(
        scaled(builder.sub(builder.add(lightness, pivot), one)), scale
    )
    factors = (shadow_factor, midtone_factor, highlight_factor)

    outputs: dict[int, tuple[int, str]] = {}
    for prefix, channel in _COLORBALANCE_RANGES:
        value = values[channel]
        for (suffix, _), factor in zip(_COLORBALANCE_WEIGHTS, factors, strict=True):
            # v += s, then v += m, then v += h: three separate roundings.
            weight = builder.const(weights[f"{prefix}{suffix}"])
            value = builder.add(value, builder.mul(weight, factor))
        clipped = builder.clipf(value, zero, one)
        outputs[channel] = (builder.mul(clipped, maximum), "lrintf_saturate_u8")
    # Alpha is copied rather than computed, so the program never stores it.
    return builder.build(outputs)


def _lower_colorbalance(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    names = {
        f"{prefix}{suffix}"
        for prefix, _ in _COLORBALANCE_RANGES
        for suffix, _ in _COLORBALANCE_WEIGHTS
    }
    options = _check_options(invocation, names | {"pl"})

    preserve = options.get("pl")
    if preserve is not None and _parse_bool(preserve.value, "pl"):
        raise LoweringError(
            "unsupported_preserve",
            "pl=1 runs preservel, whose multiply-adds are not exact, so whether "
            "the host compiler fuses them would decide output bytes",
            "pl",
        )

    weights = {
        name: _float32(
            _parse_number(options[name].value, name, -1.0, 1.0)
            if name in options
            else 0.0
        )
        for name in names
    }
    program = _colorbalance_program(weights)
    canonical = "colorbalance=" + ":".join(
        f"{prefix}{suffix}={weights[f'{prefix}{suffix}'].hex()}"
        for prefix, _ in _COLORBALANCE_RANGES
        for suffix, _ in _COLORBALANCE_WEIGHTS
    ) + ":pl=0"
    return Lowered(_expr_operations(program, invocation, index), canonical)


#: ``colorcontrast``'s contrast axes, their weights, and the channels each
#: difference is taken against.
_COLORCONTRAST_OPTIONS = ("rc", "gm", "by", "rcw", "gmw", "byw", "pl")

#: ``FLT_EPSILON``.  Upstream compares the weight sum against it to decide
#: whether to touch the frame at all, and adds it to the output lightness so
#: the division below can never be by zero.
_FLT_EPSILON = 2.0**-23


def _colorcontrast_program(values: dict[str, float], *, fused: bool) -> ExprProgram:
    """Transcribe ``vf_colorcontrast.c``'s ``PROCESS`` macro.

    Every multiply-add here scales by a user coefficient, so none of them is
    exact and each one is a place where the host compiler's contraction decides
    a byte.  ``fused`` says which way this host goes; the eighteen sites below
    are exactly the ones Clang contracts, checked against a contracted build
    over the complete ``256^3`` domain.
    """

    builder = ExprBuilder()
    zero = builder.const(0.0)
    maximum = builder.const(255.0)
    half = builder.const(0.5)

    # Loop-invariant scalars upstream computes once per slice, in float32.
    green_magenta = builder.const(_float32(values["gm"] * 0.5))
    blue_yellow = builder.const(_float32(values["by"] * 0.5))
    red_cyan = builder.const(_float32(values["rc"] * 0.5))
    green_weight = builder.const(values["gmw"])
    blue_weight = builder.const(values["byw"])
    red_weight = builder.const(values["rcw"])
    total = _float32(_float32(values["gmw"] + values["byw"]) + values["rcw"])
    scale = builder.const(_float32(1.0 / total))
    preserve = builder.const(values["pl"])
    epsilon = builder.const(_FLT_EPSILON)

    def mul_add(left: int, right: int, addend: int) -> int:
        return builder.multiply_add(left, right, addend, fused=fused)

    def subtract_product(minuend: int, left: int, right: int) -> int:
        """``minuend - left * right``, fused the way upstream's compiler fuses it."""

        if fused:
            return builder.fma(builder.neg(left), right, minuend)
        return builder.sub(minuend, builder.mul(left, right))

    # Channels enter as raw byte values here; unlike colorbalance nothing is
    # normalized, and the clamps below are against 255 rather than 1.
    red = builder.channel(0)
    green = builder.channel(1)
    blue = builder.channel(2)

    blue_red = builder.mul(builder.add(blue, red), half)
    green_blue = builder.mul(builder.add(green, blue), half)
    red_green = builder.mul(builder.add(red, green), half)

    green_difference = builder.sub(green, blue_red)
    blue_difference = builder.sub(blue, red_green)
    red_difference = builder.sub(red, green_blue)

    axes = (
        # (difference, coefficient, which channel the difference adds to)
        (green_difference, green_magenta, 1),
        (blue_difference, blue_yellow, 2),
        (red_difference, red_cyan, 0),
    )
    # Each axis pushes one channel along its difference and pulls the other two
    # back: g0/b0/r0 for green-magenta, then g1/b1/r1, then g2/b2/r2.
    shifted: list[list[int]] = []
    for difference, coefficient, toward in axes:
        row: list[int] = []
        for channel, base in enumerate((red, green, blue)):
            if channel == toward:
                row.append(mul_add(difference, coefficient, base))
            else:
                row.append(subtract_product(base, difference, coefficient))
        shifted.append(row)

    weights = (green_weight, blue_weight, red_weight)
    mixed: list[int] = []
    for channel in range(3):
        # (x0 * gmw + x1 * byw) + x2 * rcw, contracted left to right.
        first = builder.mul(shifted[1][channel], weights[1])
        partial = mul_add(shifted[0][channel], weights[0], first)
        total_terms = mul_add(shifted[2][channel], weights[2], partial)
        mixed.append(builder.clipf(builder.mul(total_terms, scale), zero, maximum))

    input_lightness = builder.add(
        builder.max3(red, green, blue), builder.min3(red, green, blue)
    )
    output_lightness = builder.add(
        builder.add(builder.max3(*mixed), builder.min3(*mixed)), epsilon
    )
    lightness_factor = builder.div(input_lightness, output_lightness)

    outputs: dict[int, tuple[int, str]] = {}
    for channel in range(3):
        restored = builder.mul(mixed[channel], lightness_factor)
        # lerpf(v0, v1, f) is v0 + (v1 - v0) * f.
        blended = mul_add(
            builder.sub(restored, mixed[channel]), preserve, mixed[channel]
        )
        outputs[channel] = (blended, "truncate_saturate_u8")
    return builder.build(outputs)


def _lower_colorcontrast(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    options = _check_options(invocation, set(_COLORCONTRAST_OPTIONS))
    values: dict[str, float] = {}
    for name in _COLORCONTRAST_OPTIONS:
        minimum = -1.0 if name in {"rc", "gm", "by"} else 0.0
        values[name] = _float32(
            _parse_number(options[name].value, name, minimum, 1.0)
            if name in options
            else 0.0
        )

    total = _float32(_float32(values["gmw"] + values["byw"]) + values["rcw"])
    canonical = "colorcontrast=" + ":".join(
        f"{name}={values[name].hex()}" for name in _COLORCONTRAST_OPTIONS
    )
    if not total > _FLT_EPSILON:
        # Upstream's slice loop is "y < slice_end && sum > FLT_EPSILON", so the
        # frame is left untouched -- which is what the defaults do.
        program = ExprBuilder().build({})
        return Lowered(
            _expr_operations(program, invocation, index), canonical + ":identity=1"
        )

    try:
        fused = target_fuses_multiply_add()
    except UnknownTargetError as error:
        raise LoweringError("unknown_target", str(error)) from error
    program = _colorcontrast_program(values, fused=fused)
    return Lowered(
        _expr_operations(program, invocation, index),
        f"{canonical}:fused_multiply_add={int(fused)}",
    )


#: ``curves``' presets, verbatim from ``curves_presets`` in ``vf_curves.c``.
#:
#: A preset only fills a component the caller left unset, and it is expanded in
#: ``curves_init`` before the tables are built, so it is exactly a default for
#: the four point strings.
_CURVES_PRESETS: dict[str, dict[str, str]] = {
    "none": {},
    "color_negative": {
        "r": "0.129/1 0.466/0.498 0.725/0",
        "g": "0.109/1 0.301/0.498 0.517/0",
        "b": "0.098/1 0.235/0.498 0.423/0",
    },
    "cross_process": {
        "r": "0/0 0.25/0.156 0.501/0.501 0.686/0.745 1/1",
        "g": "0/0 0.25/0.188 0.38/0.501 0.745/0.815 1/0.815",
        "b": "0/0 0.231/0.094 0.709/0.874 1/1",
    },
    "darker": {"master": "0/0 0.5/0.4 1/1"},
    "increase_contrast": {
        "master": "0/0 0.149/0.066 0.831/0.905 0.905/0.98 1/1"
    },
    "lighter": {"master": "0/0 0.4/0.5 1/1"},
    "linear_contrast": {"master": "0/0 0.305/0.286 0.694/0.713 1/1"},
    "medium_contrast": {"master": "0/0 0.286/0.219 0.639/0.643 1/1"},
    "negative": {"master": "0/1 1/0"},
    "strong_contrast": {
        "master": "0/0 0.301/0.196 0.592/0.6 0.686/0.737 1/1"
    },
    "vintage": {
        "r": "0/0.11 0.42/0.51 1/0.95",
        "g": "0/0 0.50/0.48 1/1",
        "b": "0/0.22 0.49/0.44 1/0.8",
    },
}

#: The preset names in declaration order, so ``preset=3`` means what it does
#: upstream, where the option is an int with named constants.
_CURVES_PRESET_ORDER = (
    "none",
    "color_negative",
    "cross_process",
    "darker",
    "increase_contrast",
    "lighter",
    "linear_contrast",
    "medium_contrast",
    "negative",
    "strong_contrast",
    "vintage",
)

#: ``curves`` gives each component two spellings for one field, so naming both
#: would silently keep whichever came last.  They are refused instead, the way
#: ``hue=h`` and ``hue=H`` are.
_CURVES_COMPONENTS = (
    ("red", "r", 0),
    ("green", "g", 1),
    ("blue", "b", 2),
    ("master", "m", 3),
)

_CURVES_INTERPOLATIONS = ("natural", "pchip")


def _curves_component_strings(
    options: dict[str, Any]
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve the four point strings the way ``curves_init`` resolves them.

    An explicit component wins over ``all``, which wins over the preset; the
    preset is the only one of the three that can set the master curve.
    """

    points: list[str | None] = [None, None, None, None]
    for long_name, short_name, slot in _CURVES_COMPONENTS:
        given = [name for name in (long_name, short_name) if name in options]
        if len(given) > 1:
            raise LoweringError(
                "conflicting_options",
                f"{long_name} and {short_name} are two spellings of one option, "
                "so naming both keeps whichever upstream parsed last",
                short_name,
            )
        if given:
            points[slot] = options[given[0]].value

    every = options.get("all")
    if every is not None:
        for slot in range(3):
            if points[slot] is None:
                points[slot] = every.value

    preset = options.get("preset")
    if preset is not None:
        name = preset.value.strip().lower()
        if name.isdigit() and int(name) < len(_CURVES_PRESET_ORDER):
            name = _CURVES_PRESET_ORDER[int(name)]
        if name not in _CURVES_PRESETS:
            raise LoweringError(
                "invalid_value",
                f"unknown curves preset {preset.value!r}; upstream accepts "
                + ", ".join(_CURVES_PRESET_ORDER),
                "preset",
            )
        defaults = _CURVES_PRESETS[name]
        for long_name, _, slot in _CURVES_COMPONENTS:
            key = "master" if slot == 3 else long_name[0]
            if points[slot] is None and key in defaults:
                points[slot] = defaults[key]

    return tuple(points)  # type: ignore[return-value]


def _lower_curves(
    invocation: FilterInvocation, index: int, layout: PixelLayout | None
) -> Lowered:
    """Lower ``curves`` into the four-channel table its slice functions apply.

    ``curves`` is channel-independent: ``dst[x + r] = graph[R][src[x + r]]``,
    with alpha copied.  All of its difficulty is in ``config_input``, which
    :mod:`lavfi_cc.curves` reproduces, so the operation this emits is the same
    ``lut8`` ``lutrgb`` emits.
    """

    names = {name for long_name, short, _ in _CURVES_COMPONENTS for name in (long_name, short)}
    # psfile and plot are accepted here only so the refusal below can say why.
    options = _check_options(
        invocation, names | {"all", "preset", "interp", "psfile", "plot"}
    )

    for rejected, reason in (
        ("psfile", "reads a Photoshop curves file, so the tables are not in the graph"),
        ("plot", "writes a Gnuplot script, which is a side effect a kernel has not got"),
    ):
        if rejected in options:
            raise LoweringError("runtime_option", f"{rejected} {reason}", rejected)

    interpolation = "natural"
    given = options.get("interp")
    if given is not None:
        interpolation = given.value.strip().lower()
        if interpolation.isdigit() and int(interpolation) < len(_CURVES_INTERPOLATIONS):
            interpolation = _CURVES_INTERPOLATIONS[int(interpolation)]
        if interpolation not in _CURVES_INTERPOLATIONS:
            raise LoweringError(
                "invalid_value",
                f"interp must be natural or pchip, got {given.value!r}",
                "interp",
            )

    def resolve_fusion() -> bool:
        # Only reached by a curve that passes exactly through a byte boundary,
        # where upstream's own output differs between hosts.
        try:
            return target_fuses_multiply_add()
        except UnknownTargetError as error:
            raise CurveError(
                f"this curve passes exactly through a byte boundary, so its "
                f"table depends on the host: {error}"
            ) from error

    strings = _curves_component_strings(options)
    tables: list[tuple[int, ...]] = []
    try:
        master = (
            None
            if strings[3] is None
            else build_curve_table(
                parse_curve_points(strings[3], "master"),
                interpolation,
                "master",
                resolve_fusion,
            )
        )
        for long_name, _, slot in _CURVES_COMPONENTS[:3]:
            points = parse_curve_points(strings[slot] or "", long_name)
            table = build_curve_table(
                points, interpolation, long_name, resolve_fusion
            )
            # config_input composes the master on top of every component, and
            # only when its point string was set at all.
            tables.append(table if master is None else compose_curves(table, master))
    except CurveError as error:
        raise LoweringError("unsupported_curve", str(error)) from error

    # Alpha is copied rather than curved: NB_COMP is three, and graph[3] is the
    # master curve, not an alpha one.
    tables.append(_identity_table())
    operation = Operation(
        "lut8", {"tables": tuple(tables)}, source_ref(index, invocation)
    )
    quantize = Operation(
        "quantize_rgba8", {"mode": "lookup_u8"}, source_ref(index, invocation)
    )
    canonical = "curves=" + ":".join(
        f"{name}=table:{_table_hash(table)}"
        for name, table in zip(("r", "g", "b", "a"), tables, strict=True)
    )
    return Lowered((operation, quantize), canonical)


Lowerer = Callable[[FilterInvocation, int, "PixelLayout | None"], Lowered]

LOWERERS: dict[str, Lowerer] = {
    "negate": _lower_negate,
    "lutrgb": _lower_lutrgb,
    "lutyuv": _lower_lutyuv,
    "eq": _lower_eq,
    "hue": _lower_hue,
    "colorlevels": _lower_colorlevels,
    "colorchannelmixer": _lower_colorchannelmixer,
    "colorbalance": _lower_colorbalance,
    "colorcontrast": _lower_colorcontrast,
    "curves": _lower_curves,
}

SUPPORTED_FILTERS = tuple(LOWERERS)


#: The 8-bit formats each filter advertises in pinned FFmpeg.
#:
#: A run may only be fused in a format that *every* filter in it accepts.
#: Otherwise FFmpeg negotiation inserts a conversion in the middle of the run
#: and one kernel would no longer be equivalent to the filters it replaced.
#:
#: These sets cover only the 8-bit formats the backend implements.  Every
#: accepted filter also advertises higher-depth formats, and ``negate``
#: advertises far more YUV formats than the three planar ones here; those are
#: deliberately out of scope, so callers must only consult this table for a
#: format in :data:`lavfi_cc.layouts.LAYOUTS`.  ``colorlevels`` and
#: ``colorchannelmixer`` additionally accept the ``0rgb``/``rgb0`` family that
#: ``negate`` and ``lutrgb`` do not, so it is outside the common subset.
#:
#: The two families barely overlap: ``negate`` advertises both, while
#: ``lutrgb``, ``colorlevels``, and ``colorchannelmixer`` are RGB-only and
#: ``lutyuv`` is YUV-only.  A run mixing the two families is therefore not
#: contiguous in one format and is refused rather than fused.
_RGB8 = frozenset(
    {"rgba", "bgra", "argb", "abgr", "rgb24", "bgr24", "gbrp", "gbrap"}
)
_YUV8 = frozenset({"yuv444p", "yuv422p", "yuv420p"})

FILTER_FORMATS: dict[str, frozenset[str]] = {
    "negate": _RGB8 | _YUV8,
    "lutrgb": _RGB8,
    "lutyuv": _YUV8,
    "eq": _YUV8,
    "hue": _YUV8,
    "colorlevels": _RGB8,
    "colorchannelmixer": _RGB8,
    "colorbalance": _RGB8,
    "colorcontrast": _RGB8,
    "curves": _RGB8,
}


def filter_supports_format(name: str, pixel_format: str) -> bool:
    """Whether *name* advertises this 8-bit format upstream.

    Only meaningful for a format in :data:`lavfi_cc.layouts.LAYOUTS`.
    """

    return pixel_format in FILTER_FORMATS.get(name, frozenset())
