"""Eligibility analysis and lowering from parsed filters to the pixel IR."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

from .expressions import ExpressionError, build_lut
from .ir import CHANNELS, Operation, PixelIR, source_ref
from .parser import (
    FilterInvocation,
    Filtergraph,
    FiltergraphSyntaxError,
    SourceSpan,
    parse_filtergraph,
)


SUPPORTED_FILTERS = ("negate", "lutrgb", "colorlevels", "colorchannelmixer")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    offset: int | None = None
    filter_index: int | None = None
    option: str | None = None

    def format(self) -> str:
        locations: list[str] = []
        if self.filter_index is not None:
            locations.append(f"filter[{self.filter_index}]")
        if self.option is not None:
            locations.append(f"option {self.option!r}")
        if self.offset is not None:
            locations.append(f"byte {self.offset}")
        where = f" ({', '.join(locations)})" if locations else ""
        return f"{self.code}: {self.message}{where}"


@dataclass(frozen=True)
class Analysis:
    source: str
    graph: Filtergraph | None
    region: tuple[int, int] | None
    diagnostics: tuple[Diagnostic, ...]
    ir: PixelIR | None
    canonical_filters: tuple[str, ...] = ()
    rewritten_filtergraph: str | None = None

    @property
    def eligible(self) -> bool:
        return self.ir is not None and not self.diagnostics

    def as_dict(self, include_ir: bool = True) -> dict[str, Any]:
        filters: list[dict[str, Any]] = []
        if self.graph is not None:
            for invocation in self.graph.filters:
                filters.append(
                    {
                        "name": invocation.name,
                        "options": [
                            {"name": option.name, "value": option.value}
                            for option in invocation.options
                        ],
                        "span": invocation.span.as_dict(),
                    }
                )
        result: dict[str, Any] = {
            "eligible": self.eligible,
            "filters": filters,
            "region": list(self.region) if self.region else None,
            "diagnostics": [diagnostic.__dict__ for diagnostic in self.diagnostics],
            "canonical_filters": list(self.canonical_filters),
            "rewritten_filtergraph": self.rewritten_filtergraph,
        }
        if include_ir:
            result["ir"] = self.ir.debug_dict() if self.ir else None
            result["canonical_ir"] = self.ir.canonical_dict() if self.ir else None
            result["plan_hash"] = self.ir.plan_hash if self.ir else None
        return result


class LoweringError(ValueError):
    def __init__(self, code: str, message: str, option: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.option = option


@dataclass(frozen=True)
class _Lowered:
    operations: tuple[Operation, ...]
    canonical: str
    removes_color_side_data: bool = False


def _format_value(invocation: FilterInvocation) -> str:
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


def _lower_negate(invocation: FilterInvocation, index: int) -> _Lowered:
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
    return _Lowered((operation, quantize), f"negate=components={ordered}")


def _lower_lutrgb(invocation: FilterInvocation, index: int) -> _Lowered:
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
        f"{channel}=table:{_table_hash(table)}" for channel, table in zip(CHANNELS, tables, strict=True)
    )
    return _Lowered((operation, quantize), canonical, removes_color_side_data=True)


def _table_hash(table: tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(table)).hexdigest()


def _lower_colorlevels(invocation: FilterInvocation, index: int) -> _Lowered:
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
    return _Lowered((operation, quantize), canonical)


def _lower_colorchannelmixer(invocation: FilterInvocation, index: int) -> _Lowered:
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
    return _Lowered((operation, quantize), canonical)


_LOWERERS = {
    "negate": _lower_negate,
    "lutrgb": _lower_lutrgb,
    "colorlevels": _lower_colorlevels,
    "colorchannelmixer": _lower_colorchannelmixer,
}


def _diagnostic_from_lowering(
    error: LoweringError, invocation: FilterInvocation, index: int
) -> Diagnostic:
    offset = invocation.span.start
    if error.option:
        option = invocation.named_options().get(error.option)
        if option is not None:
            offset = option.span.start
    return Diagnostic(error.code, error.message, offset, index, error.option)


def _find_region(graph: Filtergraph) -> tuple[tuple[int, int] | None, list[Diagnostic]]:
    inputs: list[int] = []
    malformed_formats: list[tuple[int, LoweringError]] = []
    for index, invocation in enumerate(graph.filters):
        if invocation.name != "format":
            continue
        try:
            value = _format_value(invocation)
        except LoweringError as error:
            malformed_formats.append((index, error))
            continue
        if value == "rgba":
            inputs.append(index)

    if not inputs:
        if malformed_formats:
            index, error = malformed_formats[0]
            invocation = graph.filters[index]
            return None, [_diagnostic_from_lowering(error, invocation, index)]
        return None, [
            Diagnostic(
                "missing_input_boundary",
                "no explicit format=rgba input boundary was found",
            )
        ]

    candidates: list[tuple[int, int]] = []
    unterminated: list[int] = []
    for input_index in inputs:
        output_index = next(
            (
                index
                for index in range(input_index + 1, len(graph.filters))
                if graph.filters[index].name == "format"
            ),
            None,
        )
        if output_index is None:
            unterminated.append(input_index)
        else:
            candidates.append((input_index, output_index))

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) > 1:
        return None, [
            Diagnostic(
                "ambiguous_regions",
                "multiple RGBA regions were found; the frontend requires exactly one",
            )
        ]
    if not unique_candidates:
        index = unterminated[0]
        return None, [
            Diagnostic(
                "missing_output_boundary",
                "format=rgba is not followed by an explicit output format boundary",
                graph.filters[index].span.start,
                index,
            )
        ]

    input_index, output_index = unique_candidates[0]
    if output_index == input_index + 1:
        return None, [
            Diagnostic(
                "empty_region",
                "the RGBA boundaries contain no filters to fuse",
                graph.filters[output_index].span.start,
                output_index,
            )
        ]
    try:
        _format_value(graph.filters[output_index])
    except LoweringError as error:
        return None, [
            _diagnostic_from_lowering(error, graph.filters[output_index], output_index)
        ]
    return (input_index + 1, output_index), []


def analyze_filtergraph(source: str) -> Analysis:
    """Parse, validate, and lower a complete ``-vf`` value."""

    try:
        graph = parse_filtergraph(source)
    except FiltergraphSyntaxError as error:
        return Analysis(
            source,
            None,
            None,
            (Diagnostic("syntax_error", error.message, error.offset),),
            None,
        )

    region, diagnostics = _find_region(graph)
    if diagnostics or region is None:
        return Analysis(source, graph, None, tuple(diagnostics), None)

    start, end = region
    operations: list[Operation] = [Operation("load_rgba8", {})]
    canonical: list[str] = []
    removes_color_side_data = False
    for index in range(start, end):
        invocation = graph.filters[index]
        lowerer = _LOWERERS.get(invocation.name)
        if lowerer is None:
            diagnostics.append(
                Diagnostic(
                    "unsupported_filter",
                    f"filter {invocation.name!r} is not in the supported subset "
                    f"({', '.join(SUPPORTED_FILTERS)})",
                    invocation.span.start,
                    index,
                )
            )
            continue
        try:
            lowered = lowerer(invocation, index)
        except LoweringError as error:
            diagnostics.append(_diagnostic_from_lowering(error, invocation, index))
            continue
        operations.extend(lowered.operations)
        canonical.append(lowered.canonical)
        removes_color_side_data |= lowered.removes_color_side_data

    if diagnostics:
        return Analysis(source, graph, region, tuple(diagnostics), None, tuple(canonical))

    operations.append(Operation("store_rgba8", {}))
    effects = ("remove_color_dependent_side_data",) if removes_color_side_data else ()
    ir = PixelIR(tuple(operations), effects)
    original = [invocation.raw for invocation in graph.filters]
    remove_color = int("remove_color_dependent_side_data" in effects)
    planned_filter = (
        "fused=kernel=KERNEL_PATH:kernel_root=KERNEL_ROOT:"
        f"plan_hash={ir.plan_hash}:remove_color_side_data={remove_color}"
    )
    rewritten = original[:start] + [planned_filter] + original[end:]
    return Analysis(
        source,
        graph,
        region,
        (),
        ir,
        tuple(canonical),
        ",".join(rewritten),
    )


def require_ir(source: str) -> PixelIR:
    analysis = analyze_filtergraph(source)
    if analysis.ir is None:
        details = "; ".join(diagnostic.format() for diagnostic in analysis.diagnostics)
        raise ValueError(details)
    return analysis.ir
