"""Discovery of fusible islands inside an arbitrary linear filter chain.

An *island* is a maximal run of consecutive filters that all lower into the
pixel IR.  Discovery does not require the caller to bracket the run with
explicit ``format`` filters; it tracks the pixel format along the chain instead.

Tracking the format is a correctness requirement rather than a cost model.  A
pointwise filter produces different bytes in different pixel formats -- pinned
FFmpeg's ``negate`` over ``testsrc2`` does not agree with ``format=rgba,negate``
-- so a kernel may only replace a run when the run already works in a format
the kernel implements natively.  An island in any other format is reported, not
fused, and names the format that would have to be supported to unlock it.

Each filter in an unfused run reads and writes a whole frame, so a run of ``n``
filters costs ``n`` frame passes and its kernel costs one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .filters import (
    LOWERERS,
    Lowered,
    LoweringError,
    filter_supports_format,
    format_value,
)
from .layouts import LAYOUTS
from .parser import FilterInvocation


#: Pixel formats the generated kernel reads and writes natively.
NATIVE_FORMATS = frozenset(LAYOUTS)

#: Filters that break a run but provably hand the same pixel format onward.
#:
#: These are geometric or timing filters: they move, drop, or reframe samples
#: without reinterpreting a component, so a format pinned before one is still
#: pinned after it.  ``scale`` is deliberately absent because it converts.
FORMAT_PRESERVING = frozenset(
    {
        "copy",
        "crop",
        "fifo",
        "fps",
        "hflip",
        "null",
        "pad",
        "realtime",
        "select",
        "setdar",
        "setpts",
        "setsar",
        "settb",
        "transpose",
        "trim",
        "vflip",
    }
)

#: The island already works in a format the kernel implements.
NATIVE = "native"
#: The island works in a known format the kernel does not implement yet.
UNSUPPORTED_FORMAT = "unsupported_format"
#: No ``format`` filter pins the working format, so negotiation decides it.
NEGOTIATED = "negotiated"


@dataclass(frozen=True)
class Rejection:
    """Why one filter could not join an island."""

    index: int
    name: str
    code: str
    message: str
    option: str | None = None
    offset: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "code": self.code,
            "message": self.message,
            "option": self.option,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class Island:
    """A maximal fusible run together with what fusing it would remove."""

    chain_index: int
    start: int
    end: int
    names: tuple[str, ...]
    lowered: tuple[Lowered, ...]
    working_format: str | None
    boundary: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def fusible(self) -> bool:
        """Whether a kernel may replace this run without changing its output."""

        return self.boundary == NATIVE

    @property
    def potential_eliminated_passes(self) -> int:
        """Frame passes a kernel would remove if this island's format were supported."""

        return self.length - 1

    @property
    def eliminated_passes(self) -> int:
        """Frame passes removable today."""

        return self.potential_eliminated_passes if self.fusible else 0

    @property
    def profitable(self) -> bool:
        return self.fusible and self.eliminated_passes > 0

    @property
    def blocked_reason(self) -> str | None:
        if self.fusible:
            return None
        if self.boundary == UNSUPPORTED_FORMAT:
            return (
                f"the island works in {self.working_format!r}, which the kernel does "
                "not implement natively; fusing it in a supported format would "
                "change its output"
            )
        return (
            "no format filter pins the working format, so FFmpeg negotiation "
            "chooses it and a kernel could not guarantee the same bytes"
        )

    @property
    def removes_color_side_data(self) -> bool:
        return any(item.removes_color_side_data for item in self.lowered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_index": self.chain_index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "filters": list(self.names),
            "working_format": self.working_format,
            "boundary": self.boundary,
            "fusible": self.fusible,
            "eliminated_passes": self.eliminated_passes,
            "potential_eliminated_passes": self.potential_eliminated_passes,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class ChainScan:
    """Every island and rejection found in one chain."""

    chain_index: int
    islands: tuple[Island, ...]
    rejections: tuple[Rejection, ...]
    filter_count: int


def _boundary_for(working_format: str | None) -> str:
    if working_format in NATIVE_FORMATS:
        return NATIVE
    if working_format is None:
        return NEGOTIATED
    return UNSUPPORTED_FORMAT


def _rejection(
    index: int, invocation: FilterInvocation, error: LoweringError
) -> Rejection:
    offset = invocation.span.start
    if error.option:
        option = invocation.named_options().get(error.option)
        if option is not None:
            offset = option.span.start
    return Rejection(
        index, invocation.name, error.code, error.message, error.option, offset
    )


def scan_chain(
    filters: Sequence[FilterInvocation],
    chain_index: int = 0,
    *,
    entry_format: str | None = None,
) -> ChainScan:
    """Find every fusible island in one chain, tracking the pixel format.

    ``entry_format`` is the format entering the chain when the caller can prove
    it -- feeding raw RGBA8 frames, for instance.  Leave it unset when FFmpeg
    negotiation decides, which is the case for a decoded input.
    """

    islands: list[Island] = []
    rejections: list[Rejection] = []

    current_format: str | None = entry_format
    run_start: int | None = None
    run_format: str | None = None
    run: list[Lowered] = []
    run_names: list[str] = []

    def close_run(end: int) -> None:
        nonlocal run_start
        if run_start is None:
            return
        islands.append(
            Island(
                chain_index,
                run_start,
                end,
                tuple(run_names),
                tuple(run),
                run_format,
                _boundary_for(run_format),
            )
        )
        run_start = None
        run.clear()
        run_names.clear()

    for index, invocation in enumerate(filters):
        if invocation.name == "format":
            close_run(index)
            try:
                current_format = format_value(invocation)
            except LoweringError as error:
                rejections.append(_rejection(index, invocation, error))
                current_format = None
            continue

        lowerer = LOWERERS.get(invocation.name)
        if lowerer is None:
            close_run(index)
            rejections.append(
                Rejection(
                    index,
                    invocation.name,
                    "unsupported_filter",
                    f"filter {invocation.name!r} is not in the supported subset",
                    None,
                    invocation.span.start,
                )
            )
            if invocation.name not in FORMAT_PRESERVING:
                # Any other filter may change the format arbitrarily.
                current_format = None
            continue

        # Only checkable for a format the backend implements; for anything else
        # the island is already not fusible and the run stays intact for
        # reporting.
        if current_format in NATIVE_FORMATS and not filter_supports_format(
            invocation.name, current_format
        ):
            # FFmpeg would convert into a format this filter accepts, so the
            # run is not actually contiguous in one format.
            close_run(index)
            rejections.append(
                Rejection(
                    index,
                    invocation.name,
                    "format_not_advertised",
                    f"{invocation.name} does not advertise {current_format!r}, "
                    "so FFmpeg converts around it",
                    None,
                    invocation.span.start,
                )
            )
            current_format = None
            continue

        try:
            # An undecided format leaves the layout-dependent checks to the
            # caller, which already refuses to fuse a run it cannot name a
            # format for.
            layout = LAYOUTS.get(current_format or "")
            lowered = lowerer(invocation, index, layout)
        except LoweringError as error:
            close_run(index)
            rejections.append(_rejection(index, invocation, error))
            # A rejected filter is still pointwise, so the format is unchanged.
            continue

        if run_start is None:
            run_start = index
            run_format = current_format
        run.append(lowered)
        run_names.append(invocation.name)

    close_run(len(filters))
    return ChainScan(chain_index, tuple(islands), tuple(rejections), len(filters))


def scan_chains(chains: Iterable[Sequence[FilterInvocation]]) -> tuple[ChainScan, ...]:
    return tuple(scan_chain(filters, index) for index, filters in enumerate(chains))


def select_islands(scans: Iterable[ChainScan]) -> tuple[Island, ...]:
    """Return the islands a kernel may replace today, in chain and source order."""

    return tuple(
        island for scan in scans for island in scan.islands if island.profitable
    )
