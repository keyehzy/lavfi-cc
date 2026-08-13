"""Eligibility analysis and lowering from parsed filters to the pixel IR.

Two discovery modes share one lowering layer.  The default mode requires the
caller to bracket the fusible run with explicit ``format`` filters, which keeps
the fused region exactly where the author put it.  Auto-island mode instead
asks :mod:`lavfi_cc.islands` to find fusible runs anywhere in the chain and
fuses each run that removes more frame passes than it costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .filters import (
    LOWERERS,
    SUPPORTED_FILTERS,
    Lowered,
    LoweringError,
    filter_supports_format,
    format_value,
)
from .ir import Operation, PixelIR
from .layouts import DEFAULT_LAYOUT, LAYOUTS
from .islands import Island, scan_chain, select_islands
from .parser import (
    FilterInvocation,
    Filtergraph,
    FiltergraphSyntaxError,
    parse_filtergraph,
)


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
class IslandPlan:
    """One fusible region together with the IR it lowers to."""

    region: tuple[int, int]
    ir: PixelIR
    canonical_filters: tuple[str, ...]
    island: Island | None = None

    @property
    def boundary(self) -> str:
        return self.island.boundary if self.island is not None else "native"

    @property
    def eliminated_passes(self) -> int:
        if self.island is not None:
            return self.island.eliminated_passes
        return (self.region[1] - self.region[0]) - 1

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "region": list(self.region),
            "plan_hash": self.ir.plan_hash,
            "canonical_filters": list(self.canonical_filters),
            "boundary": self.boundary,
            "eliminated_passes": self.eliminated_passes,
        }
        if self.island is not None:
            value["island"] = self.island.as_dict()
        return value


@dataclass(frozen=True)
class Analysis:
    source: str
    graph: Filtergraph | None
    diagnostics: tuple[Diagnostic, ...] = ()
    plans: tuple[IslandPlan, ...] = ()
    rewritten_filtergraph: str | None = None
    auto_islands: bool = False

    @property
    def eligible(self) -> bool:
        return bool(self.plans) and not self.diagnostics

    @property
    def plan(self) -> IslandPlan | None:
        """The single plan, when this analysis produced exactly one."""

        return self.plans[0] if len(self.plans) == 1 else None

    @property
    def ir(self) -> PixelIR | None:
        plan = self.plan
        return plan.ir if plan is not None else None

    @property
    def region(self) -> tuple[int, int] | None:
        plan = self.plan
        return plan.region if plan is not None else None

    @property
    def canonical_filters(self) -> tuple[str, ...]:
        plan = self.plan
        return plan.canonical_filters if plan is not None else ()

    @property
    def eliminated_passes(self) -> int:
        return sum(plan.eliminated_passes for plan in self.plans)

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
            "auto_islands": self.auto_islands,
            "islands": [plan.as_dict() for plan in self.plans],
            "eliminated_passes": self.eliminated_passes,
        }
        if include_ir:
            result["ir"] = self.ir.debug_dict() if self.ir else None
            result["canonical_ir"] = self.ir.canonical_dict() if self.ir else None
            result["plan_hash"] = self.ir.plan_hash if self.ir else None
        return result


def _diagnostic_from_lowering(
    error: LoweringError, invocation: FilterInvocation, index: int
) -> Diagnostic:
    offset = invocation.span.start
    if error.option:
        option = invocation.named_options().get(error.option)
        if option is not None:
            offset = option.span.start
    return Diagnostic(error.code, error.message, offset, index, error.option)


def _find_region(
    graph: Filtergraph,
) -> tuple[tuple[int, int] | None, str, list[Diagnostic]]:
    """Find the one explicitly bracketed region and the layout it runs in."""

    inputs: list[tuple[int, str]] = []
    malformed_formats: list[tuple[int, LoweringError]] = []
    for index, invocation in enumerate(graph.filters):
        if invocation.name != "format":
            continue
        try:
            value = format_value(invocation)
        except LoweringError as error:
            malformed_formats.append((index, error))
            continue
        if value in LAYOUTS:
            inputs.append((index, value))

    if not inputs:
        if malformed_formats:
            index, error = malformed_formats[0]
            invocation = graph.filters[index]
            return None, DEFAULT_LAYOUT, [
                _diagnostic_from_lowering(error, invocation, index)
            ]
        return None, DEFAULT_LAYOUT, [
            Diagnostic(
                "missing_input_boundary",
                "no explicit format=rgba input boundary was found",
            )
        ]

    candidates: list[tuple[int, int, str]] = []
    unterminated: list[int] = []
    for input_index, layout in inputs:
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
            candidates.append((input_index, output_index, layout))

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) > 1:
        return None, DEFAULT_LAYOUT, [
            Diagnostic(
                "ambiguous_regions",
                "multiple RGBA regions were found; the frontend requires exactly one",
            )
        ]
    if not unique_candidates:
        index = unterminated[0]
        return None, DEFAULT_LAYOUT, [
            Diagnostic(
                "missing_output_boundary",
                "format=rgba is not followed by an explicit output format boundary",
                graph.filters[index].span.start,
                index,
            )
        ]

    input_index, output_index, layout = unique_candidates[0]
    if output_index == input_index + 1:
        return None, layout, [
            Diagnostic(
                "empty_region",
                "the RGBA boundaries contain no filters to fuse",
                graph.filters[output_index].span.start,
                output_index,
            )
        ]
    try:
        format_value(graph.filters[output_index])
    except LoweringError as error:
        return None, layout, [
            _diagnostic_from_lowering(error, graph.filters[output_index], output_index)
        ]
    return (input_index + 1, output_index), layout, []


def _planned_filter(ir: PixelIR) -> str:
    remove_color = int("remove_color_dependent_side_data" in ir.metadata_effects)
    return (
        "fused=kernel=KERNEL_PATH:kernel_root=KERNEL_ROOT:"
        f"plan_hash={ir.plan_hash}:remove_color_side_data={remove_color}"
    )


def build_island_ir(
    lowered: tuple[Lowered, ...], layout: str = DEFAULT_LAYOUT
) -> PixelIR:
    operations: list[Operation] = [Operation("load_rgba8", {})]
    removes_color_side_data = False
    for item in lowered:
        operations.extend(item.operations)
        removes_color_side_data |= item.removes_color_side_data
    operations.append(Operation("store_rgba8", {}))
    effects = ("remove_color_dependent_side_data",) if removes_color_side_data else ()
    return PixelIR(tuple(operations), effects, layout=layout)


def rewrite_with(
    graph: Filtergraph, plans: tuple[IslandPlan, ...], replacements: list[str]
) -> str:
    """Replace each planned region with its replacement, keeping the rest."""

    if len(plans) != len(replacements):
        raise ValueError("each plan needs exactly one replacement")
    original = [invocation.raw for invocation in graph.filters]
    output: list[str] = []
    position = 0
    for plan, replacement in zip(plans, replacements, strict=True):
        start, end = plan.region
        output.extend(original[position:start])
        output.append(replacement)
        position = end
    output.extend(original[position:])
    return ",".join(output)


def _analyze_explicit_region(source: str, graph: Filtergraph) -> Analysis:
    region, layout, diagnostics = _find_region(graph)
    if diagnostics or region is None:
        return Analysis(source, graph, tuple(diagnostics))

    start, end = region
    lowered: list[Lowered] = []
    canonical: list[str] = []
    for index in range(start, end):
        invocation = graph.filters[index]
        lowerer = LOWERERS.get(invocation.name)
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
        if not filter_supports_format(invocation.name, layout):
            diagnostics.append(
                Diagnostic(
                    "format_not_advertised",
                    f"{invocation.name} does not advertise {layout!r}, so FFmpeg "
                    "converts around it",
                    invocation.span.start,
                    index,
                )
            )
            continue
        try:
            item = lowerer(invocation, index, LAYOUTS[layout])
        except LoweringError as error:
            diagnostics.append(_diagnostic_from_lowering(error, invocation, index))
            continue
        lowered.append(item)
        canonical.append(item.canonical)

    if diagnostics:
        return Analysis(source, graph, tuple(diagnostics))

    ir = build_island_ir(tuple(lowered), layout)
    plan = IslandPlan(region, ir, tuple(canonical))
    return Analysis(
        source,
        graph,
        (),
        (plan,),
        rewrite_with(graph, (plan,), [_planned_filter(ir)]),
    )


def _analyze_islands(
    source: str, graph: Filtergraph, entry_format: str | None
) -> Analysis:
    scan = scan_chain(graph.filters, entry_format=entry_format)
    selected = select_islands((scan,))
    if not selected:
        best = max(scan.islands, key=lambda item: item.length, default=None)
        if best is None:
            detail = "no filter in the chain lowers into the pixel IR"
        elif best.blocked_reason is not None:
            detail = (
                f"the longest island spans {best.length} filter"
                f"{'s' if best.length != 1 else ''} but cannot be fused: "
                f"{best.blocked_reason}"
            )
        else:
            detail = (
                f"the longest island spans {best.length} filter"
                f"{'s' if best.length != 1 else ''}, which removes no frame pass"
            )
        diagnostics = [Diagnostic("no_profitable_island", detail)]
        diagnostics.extend(
            Diagnostic(
                rejection.code,
                rejection.message,
                rejection.offset,
                rejection.index,
                rejection.option,
            )
            for rejection in scan.rejections
        )
        return Analysis(source, graph, tuple(diagnostics), auto_islands=True)

    plans = tuple(
        IslandPlan(
            (island.start, island.end),
            build_island_ir(island.lowered, island.working_format or DEFAULT_LAYOUT),
            tuple(item.canonical for item in island.lowered),
            island,
        )
        for island in selected
    )
    rewritten = rewrite_with(
        graph, plans, [_planned_filter(plan.ir) for plan in plans]
    )
    return Analysis(source, graph, (), plans, rewritten, auto_islands=True)


def analyze_filtergraph(
    source: str,
    *,
    auto_islands: bool = False,
    entry_format: str | None = None,
) -> Analysis:
    """Parse, validate, and lower a complete ``-vf`` value.

    With ``auto_islands`` the fusible runs are discovered anywhere in the chain
    and kept when they already work in a format the kernel implements natively.
    Otherwise the run must be bracketed by an explicit ``format=rgba`` and a
    following ``format`` filter.  ``entry_format`` names the format entering the
    chain when the caller can prove it, such as a raw RGBA8 frame stream.
    """

    try:
        graph = parse_filtergraph(source)
    except FiltergraphSyntaxError as error:
        return Analysis(
            source,
            None,
            (Diagnostic("syntax_error", error.message, error.offset),),
            auto_islands=auto_islands,
        )

    if auto_islands:
        return _analyze_islands(source, graph, entry_format)
    return _analyze_explicit_region(source, graph)


def require_ir(
    source: str, *, auto_islands: bool = False, entry_format: str | None = None
) -> PixelIR:
    analysis = analyze_filtergraph(
        source, auto_islands=auto_islands, entry_format=entry_format
    )
    if analysis.ir is None:
        details = "; ".join(diagnostic.format() for diagnostic in analysis.diagnostics)
        raise ValueError(details)
    return analysis.ir
