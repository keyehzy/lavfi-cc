"""Analysis-only scanner for arbitrary, real-world filtergraphs.

The scanner never compiles, loads, or runs anything.  It parses whatever it is
given -- several chains, link labels, unsupported filters, options this
compiler cannot read -- and answers two questions for each graph: which runs a
kernel could replace today, and what is standing in the way of the rest.

The second answer is the useful one for planning.  Every island that cannot be
fused is attributed to a concrete blocker (an unsupported filter, an option
outside the accepted subset, or a working format the kernel does not implement)
and the passes it would have removed are counted separately, so the aggregate
report ranks blockers by the frame passes they are actually costing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .frontend import build_island_ir
from .islands import ChainScan, Island, NEGOTIATED, UNSUPPORTED_FORMAT, scan_chain
from .parser import FiltergraphSyntaxError, parse_filtergraph_script
from .passes import optimize_ir


@dataclass(frozen=True)
class IslandReport:
    """One island with the stage counts a kernel would produce for it."""

    island: Island
    stages: int
    optimized_stages: int

    @property
    def fusible(self) -> bool:
        return self.island.fusible

    def as_dict(self) -> dict[str, Any]:
        value = self.island.as_dict()
        value["stages"] = self.stages
        value["optimized_stages"] = self.optimized_stages
        return value


@dataclass(frozen=True)
class GraphScan:
    """Everything the scanner learned about one filtergraph."""

    source: str
    error: str | None = None
    chains: tuple[ChainScan, ...] = ()
    islands: tuple[IslandReport, ...] = ()

    @property
    def parsed(self) -> bool:
        return self.error is None

    @property
    def filter_count(self) -> int:
        return sum(chain.filter_count for chain in self.chains)

    @property
    def fusible_islands(self) -> tuple[IslandReport, ...]:
        return tuple(report for report in self.islands if report.fusible)

    @property
    def eliminated_passes(self) -> int:
        return sum(report.island.eliminated_passes for report in self.islands)

    @property
    def potential_eliminated_passes(self) -> int:
        return sum(
            report.island.potential_eliminated_passes for report in self.islands
        )

    @property
    def blocked_passes(self) -> int:
        return self.potential_eliminated_passes - self.eliminated_passes

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parsed": self.parsed,
            "error": self.error,
            "filters": self.filter_count,
            "islands": [report.as_dict() for report in self.islands],
            "eliminated_passes": self.eliminated_passes,
            "potential_eliminated_passes": self.potential_eliminated_passes,
            "blocked_passes": self.blocked_passes,
            "rejections": [
                rejection.as_dict()
                for chain in self.chains
                for rejection in chain.rejections
            ],
        }


@dataclass
class ScanSummary:
    """Totals and blocker rankings across a corpus of graphs."""

    graphs: int = 0
    parsed: int = 0
    filters: int = 0
    islands: int = 0
    fusible_islands: int = 0
    graphs_with_fusible_island: int = 0
    eliminated_passes: int = 0
    potential_eliminated_passes: int = 0
    unsupported_filters: Counter[str] = field(default_factory=Counter)
    rejected_options: Counter[str] = field(default_factory=Counter)
    blocking_formats: Counter[str] = field(default_factory=Counter)
    #: Frame passes each blocker is withholding, not just how often it appears.
    blocked_passes_by_format: Counter[str] = field(default_factory=Counter)

    @property
    def blocked_passes(self) -> int:
        return self.potential_eliminated_passes - self.eliminated_passes

    def as_dict(self) -> dict[str, Any]:
        return {
            "graphs": self.graphs,
            "parsed": self.parsed,
            "failed": self.graphs - self.parsed,
            "filters": self.filters,
            "islands": self.islands,
            "fusible_islands": self.fusible_islands,
            "graphs_with_fusible_island": self.graphs_with_fusible_island,
            "eliminated_passes": self.eliminated_passes,
            "potential_eliminated_passes": self.potential_eliminated_passes,
            "blocked_passes": self.blocked_passes,
            "unsupported_filters": dict(self.unsupported_filters.most_common()),
            "rejected_options": dict(self.rejected_options.most_common()),
            "blocking_formats": dict(self.blocking_formats.most_common()),
            "blocked_passes_by_format": dict(
                self.blocked_passes_by_format.most_common()
            ),
        }


def _report_island(island: Island) -> IslandReport:
    stages = len(island.lowered)
    try:
        ir = build_island_ir(island.lowered)
        optimized = optimize_ir(ir).ir
        # The IR body alternates transform and quantize operations.
        optimized_stages = (len(optimized.operations) - 2) // 2
    except Exception:  # pragma: no cover - analysis must never fail a scan
        optimized_stages = stages
    return IslandReport(island, stages, optimized_stages)


def scan_graph(source: str, *, entry_format: str | None = None) -> GraphScan:
    """Scan one filtergraph without compiling or running anything."""

    try:
        script = parse_filtergraph_script(source)
    except FiltergraphSyntaxError as error:
        return GraphScan(source, str(error))

    chains = tuple(
        scan_chain(chain.filters, index, entry_format=entry_format)
        for index, chain in enumerate(script.chains)
    )
    islands = tuple(
        _report_island(island) for chain in chains for island in chain.islands
    )
    return GraphScan(source, None, chains, islands)


def summarize(scans: Iterable[GraphScan]) -> ScanSummary:
    """Aggregate scans, ranking blockers by the frame passes they withhold."""

    summary = ScanSummary()
    for scan in scans:
        summary.graphs += 1
        if not scan.parsed:
            continue
        summary.parsed += 1
        summary.filters += scan.filter_count
        summary.islands += len(scan.islands)
        summary.fusible_islands += len(scan.fusible_islands)
        if scan.fusible_islands:
            summary.graphs_with_fusible_island += 1
        summary.eliminated_passes += scan.eliminated_passes
        summary.potential_eliminated_passes += scan.potential_eliminated_passes

        for chain in scan.chains:
            for rejection in chain.rejections:
                if rejection.code == "unsupported_filter":
                    summary.unsupported_filters[rejection.name] += 1
                else:
                    label = f"{rejection.name}:{rejection.code}"
                    summary.rejected_options[label] += 1

        for report in scan.islands:
            island = report.island
            if island.fusible:
                continue
            if island.boundary == UNSUPPORTED_FORMAT:
                key = island.working_format or "unknown"
            elif island.boundary == NEGOTIATED:
                key = NEGOTIATED
            else:  # pragma: no cover - every non-fusible island is one of these
                continue
            summary.blocking_formats[key] += 1
            summary.blocked_passes_by_format[key] += (
                island.potential_eliminated_passes
            )
    return summary


def load_graphs(lines: Iterable[str]) -> list[str]:
    """Read one filtergraph per line, ignoring blanks and ``#`` comments."""

    graphs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        graphs.append(stripped)
    return graphs


def render_text(scans: Sequence[GraphScan], summary: ScanSummary, *, verbose: bool) -> str:
    lines: list[str] = []
    if verbose:
        for scan in scans:
            lines.append(f"graph: {scan.source}")
            if not scan.parsed:
                lines.append(f"  unparsed: {scan.error}")
                lines.append("")
                continue
            if not scan.islands:
                lines.append("  no island lowers into the pixel IR")
            for report in scan.islands:
                island = report.island
                where = f"chain {island.chain_index} filters [{island.start}:{island.end}]"
                names = ",".join(island.names)
                if island.fusible:
                    lines.append(
                        f"  fusible   {where} {names} "
                        f"({report.stages} -> {report.optimized_stages} stage"
                        f"{'s' if report.optimized_stages != 1 else ''}, "
                        f"removes {island.eliminated_passes} pass"
                        f"{'es' if island.eliminated_passes != 1 else ''})"
                    )
                else:
                    lines.append(
                        f"  blocked   {where} {names} "
                        f"(would remove {island.potential_eliminated_passes} pass"
                        f"{'es' if island.potential_eliminated_passes != 1 else ''})"
                    )
                    lines.append(f"            {island.blocked_reason}")
            lines.append("")

    lines.append("Summary")
    lines.append(f"  graphs scanned:            {summary.graphs}")
    if summary.graphs != summary.parsed:
        lines.append(f"  graphs unparsed:           {summary.graphs - summary.parsed}")
    lines.append(f"  filters seen:              {summary.filters}")
    lines.append(f"  islands found:             {summary.islands}")
    lines.append(f"  islands fusible today:     {summary.fusible_islands}")
    lines.append(
        f"  graphs with an island:     {summary.graphs_with_fusible_island}"
    )
    lines.append(f"  frame passes eliminated:   {summary.eliminated_passes}")
    lines.append(f"  frame passes blocked:      {summary.blocked_passes}")

    if summary.blocked_passes_by_format:
        lines.append("  blocked passes by working format:")
        for name, passes in summary.blocked_passes_by_format.most_common():
            islands = summary.blocking_formats[name]
            lines.append(
                f"    {name:<12} {passes} pass"
                f"{'es' if passes != 1 else ''} across {islands} island"
                f"{'s' if islands != 1 else ''}"
            )
    if summary.unsupported_filters:
        lines.append("  most common unsupported filters:")
        for name, count in summary.unsupported_filters.most_common(10):
            lines.append(f"    {name:<20} {count}")
    if summary.rejected_options:
        lines.append("  most common rejections inside the subset:")
        for name, count in summary.rejected_options.most_common(10):
            lines.append(f"    {name:<32} {count}")
    return "\n".join(lines)
