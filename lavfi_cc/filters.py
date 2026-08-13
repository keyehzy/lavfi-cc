"""Per-filter eligibility checks and lowering into pixel-IR operations.

Each entry in :data:`LOWERERS` turns one parsed filter invocation into the
operations it contributes, or raises :class:`LoweringError` explaining exactly
why that invocation is outside the accepted subset.  Keeping this layer free of
graph-level concerns lets both the fusion frontend and the analysis-only
scanner ask the same question about a filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Callable

from .expressions import ExpressionError, build_lut
from .ir import CHANNELS, Operation, source_ref
from .layouts import PixelLayout
from .parser import FilterInvocation


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


def _identity_table() -> tuple[int, ...]:
    return tuple(range(256))


def _table_hash(table: tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(table)).hexdigest()


def _lower_negate(invocation: FilterInvocation, index: int) -> Lowered:
    options = _check_options(invocation, {"components", "negate_alpha"})
    components = {"r", "g", "b"}
    if "components" in options:
        raw_components = options["components"].value
        tokens = raw_components.split("+")
        if not tokens or any(token not in CHANNELS for token in tokens):
            raise LoweringError(
                "invalid_components",
                "components must be a + separated combination of r, g, b, and a",
                "components",
            )
        if len(set(tokens)) != len(tokens):
            raise LoweringError(
                "invalid_components", "components must not contain duplicates", "components"
            )
        components = set(tokens)
    if "negate_alpha" in options:
        # In pinned FFmpeg this legacy plane option has no alpha effect on packed RGBA.
        _parse_bool(options["negate_alpha"].value, "negate_alpha")

    identity = _identity_table()
    inverse = tuple(255 - value for value in range(256))
    tables = tuple(inverse if channel in components else identity for channel in CHANNELS)
    operation = Operation("lut8", {"tables": tables}, source_ref(index, invocation))
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "lookup_u8"},
        source_ref(index, invocation),
    )
    ordered = "+".join(channel for channel in CHANNELS if channel in components)
    return Lowered((operation, quantize), f"negate=components={ordered}")


def _lower_lutrgb(invocation: FilterInvocation, index: int) -> Lowered:
    options = _check_options(invocation, set(CHANNELS))
    expressions = {channel: "clipval" for channel in CHANNELS}
    expressions.update({name: option.value for name, option in options.items()})
    tables: list[tuple[int, ...]] = []
    for channel in CHANNELS:
        try:
            tables.append(build_lut(expressions[channel]))
        except ExpressionError as error:
            raise LoweringError(
                "unsupported_expression", str(error), channel
            ) from error
    operation = Operation("lut8", {"tables": tuple(tables)}, source_ref(index, invocation))
    quantize = Operation(
        "quantize_rgba8",
        {"mode": "truncate_toward_zero_then_saturate"},
        source_ref(index, invocation),
    )
    canonical = "lutrgb=" + ":".join(
        f"{channel}=table:{_table_hash(table)}"
        for channel, table in zip(CHANNELS, tables, strict=True)
    )
    return Lowered((operation, quantize), canonical, removes_color_side_data=True)


def _lower_colorlevels(invocation: FilterInvocation, index: int) -> Lowered:
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


def _lower_colorchannelmixer(invocation: FilterInvocation, index: int) -> Lowered:
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


Lowerer = Callable[[FilterInvocation, int], Lowered]

LOWERERS: dict[str, Lowerer] = {
    "negate": _lower_negate,
    "lutrgb": _lower_lutrgb,
    "colorlevels": _lower_colorlevels,
    "colorchannelmixer": _lower_colorchannelmixer,
}

SUPPORTED_FILTERS = tuple(LOWERERS)


#: The packed 8-bit RGB formats each filter advertises in pinned FFmpeg.
#:
#: A run may only be fused in a format that *every* filter in it accepts.
#: Otherwise FFmpeg negotiation inserts a conversion in the middle of the run
#: and one kernel would no longer be equivalent to the filters it replaced.
#:
#: These sets cover only the packed 8-bit RGB formats, which are the ones a
#: kernel can currently run in.  Every accepted filter also advertises planar
#: and higher-depth formats, and three of them advertise YUV; those are
#: deliberately out of scope here, so callers must only consult this table for
#: a format the backend actually implements.  ``colorlevels`` and
#: ``colorchannelmixer`` additionally accept the ``0rgb``/``rgb0`` family that
#: ``negate`` and ``lutrgb`` do not, so it is outside the common subset.
_RGB8 = frozenset(
    {"rgba", "bgra", "argb", "abgr", "rgb24", "bgr24", "gbrp", "gbrap"}
)

RGB8_FILTER_FORMATS: dict[str, frozenset[str]] = {
    "negate": _RGB8,
    "lutrgb": _RGB8,
    "colorlevels": _RGB8,
    "colorchannelmixer": _RGB8,
}


def filter_supports_rgb8(name: str, pixel_format: str) -> bool:
    """Whether *name* advertises this packed 8-bit RGB format upstream.

    Only meaningful for a format in :data:`lavfi_cc.layouts.LAYOUTS`.
    """

    return pixel_format in RGB8_FILTER_FORMATS.get(name, frozenset())


def _validate_negate_for_layout(
    invocation: FilterInvocation, layout: PixelLayout
) -> None:
    """Check the two ways ``negate`` depends on the pixel format.

    Upstream validates an explicit component mask against the format at
    configuration time and fails the graph outright; the default mask skips
    that check.  Separately, ``negate_alpha`` sets a *plane* mask, which packed
    RGB ignores in favour of its component mask but planar RGB obeys, so the
    legacy option really does negate alpha on ``gbrap``.
    """

    options = invocation.named_options()
    components = options.get("components")
    if components is not None and not layout.has_alpha:
        if "a" in components.value.split("+"):
            raise LoweringError(
                "component_not_available",
                f"negate cannot select alpha in {layout.name!r}, which has no "
                "alpha channel; upstream fails to configure this graph",
                "components",
            )

    negate_alpha = options.get("negate_alpha")
    if (
        negate_alpha is not None
        and components is None
        and layout.planar
        and layout.has_alpha
        and _parse_bool(negate_alpha.value, "negate_alpha")
    ):
        raise LoweringError(
            "planar_negate_alpha",
            f"negate_alpha sets a plane mask, so it negates alpha in "
            f"{layout.name!r} unlike in packed RGB; spell the intent as "
            "components=r+g+b+a instead",
            "negate_alpha",
        )


LayoutValidator = Callable[[FilterInvocation, PixelLayout], None]

#: Extra checks that only become decidable once the working format is known.
LAYOUT_VALIDATORS: dict[str, LayoutValidator] = {
    "negate": _validate_negate_for_layout,
}


def validate_for_layout(invocation: FilterInvocation, layout: PixelLayout) -> None:
    """Raise :class:`LoweringError` if this filter cannot run in this layout."""

    validator = LAYOUT_VALIDATORS.get(invocation.name)
    if validator is not None:
        validator(invocation, layout)
