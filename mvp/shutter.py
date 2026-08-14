"""Accelerated CPU color grading for Shutter Encoder exports.

This adapter deliberately accepts less than FFmpeg's complete
``-filter_complex`` language.  Its execution path is restricted to the one
linear ``[0:v]... [out]`` chain emitted by Shutter for an ordinary export.  It
normalizes Shutter's documented positional grading options, asks the pinned
FFmpeg build which format each compatible run negotiated, pins only that
observed format, and delegates lowering and compilation to :mod:`lavfi_cc`.

Any failure runs the original FFmpeg command unless strict mode was requested.
The original command is never run with a partly normalized or partly rewritten
graph.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

from lavfi_cc.cache import CacheError, KernelCache
from lavfi_cc.ffmpeg import (
    FFmpegIntegrationError,
    FiltergraphArgument,
    fused_filter,
    resolve_ffmpeg,
    rewrite_filtergraph_islands,
)
from lavfi_cc.filters import FILTER_FORMATS, LOWERERS, LoweringError
from lavfi_cc.frontend import Analysis, analyze_filtergraph
from lavfi_cc.layouts import LAYOUTS
from lavfi_cc.native import NativeError, compile_kernel
from lavfi_cc.parser import (
    FilterInvocation,
    FiltergraphSyntaxError,
    parse_filtergraph,
    parse_filtergraph_script,
)


MVP_NAME = "Accelerated CPU color grading for Shutter Encoder exports."
EXPECTED_FFMPEG_VERSION = "n8.1.2"


class ShutterIntegrationError(RuntimeError):
    """The adapter cannot prove that a Shutter graph is safe to rewrite."""


@dataclass(frozen=True)
class CandidateIsland:
    """A maximal consecutive run with at least one common native format."""

    start: int
    end: int
    names: tuple[str, ...]
    compatible_formats: frozenset[str]


@dataclass(frozen=True)
class FormatPin:
    """A format boundary justified by a result from pinned FFmpeg."""

    start: int
    end: int
    pixel_format: str
    filters: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "region": [self.start, self.end],
            "pixel_format": self.pixel_format,
            "filters": list(self.filters),
        }


@dataclass(frozen=True)
class PreparedGraph:
    """A normalized, verified graph ready for kernel materialization."""

    source: str
    normalized_chain: str
    pinned_chain: str
    pins: tuple[FormatPin, ...]
    analysis: Analysis
    probe_scope: str

    @property
    def placeholder_filtergraph(self) -> str:
        assert self.analysis.rewritten_filtergraph is not None
        return f"[0:v]{self.analysis.rewritten_filtergraph}[out]"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": MVP_NAME,
            "source": self.source,
            "normalized_chain": self.normalized_chain,
            "pinned_chain": self.pinned_chain,
            "format_pins": [pin.as_dict() for pin in self.pins],
            "format_probe_scope": self.probe_scope,
            "islands": [plan.as_dict() for plan in self.analysis.plans],
            "eliminated_passes": self.analysis.eliminated_passes,
            "rewritten_filtergraph": self.placeholder_filtergraph,
        }


# These are the positional forms emitted by the pinned Shutter source.  The
# names are FFmpeg's AVOption order, so more than the one currently emitted by
# Shutter can still be normalized without guessing.
_SHUTTER_POSITIONAL_OPTIONS: dict[str, tuple[str, ...]] = {
    "colortemperature": ("temperature", "mix", "pl"),
    "vibrance": (
        "intensity",
        "rbal",
        "gbal",
        "bbal",
        "rlum",
        "glum",
        "blum",
        "alternate",
    ),
}

_VERSION = re.compile(r"^ffmpeg version (\S+)", re.MULTILINE)
_PROBE_FRAME = re.compile(
    r"\[showinfo@lavfi_cc_probe_(\d+)\s+@[^\]]*\].*?"
    r"\bn:\s*0\b.*?\bfmt:([A-Za-z0-9_]+)"
)


def find_filter_complex_argument(arguments: Sequence[str]) -> FiltergraphArgument:
    """Find exactly one separate ``-filter_complex`` argument."""

    matches: list[FiltergraphArgument] = []
    for index, argument in enumerate(arguments):
        if argument != "-filter_complex":
            continue
        if index + 1 >= len(arguments):
            raise ShutterIntegrationError("-filter_complex is missing its graph")
        matches.append(FiltergraphArgument(index, index + 1, arguments[index + 1]))
    if not matches:
        raise ShutterIntegrationError("no -filter_complex graph was found")
    if len(matches) != 1:
        raise ShutterIntegrationError(
            "multiple -filter_complex graphs are ambiguous; exactly one is required"
        )
    return matches[0]


def _linear_shutter_filters(source: str) -> tuple[FilterInvocation, ...]:
    try:
        script = parse_filtergraph_script(source)
    except FiltergraphSyntaxError as error:
        raise ShutterIntegrationError(str(error)) from error
    if len(script.chains) != 1:
        raise ShutterIntegrationError(
            "only one linear video chain is supported; audio or branched chains "
            "remain on ordinary FFmpeg"
        )
    filters = script.chains[0].filters
    if not filters:
        raise ShutterIntegrationError("the video chain is empty")
    for index, invocation in enumerate(filters):
        expected_inputs = ("0:v",) if index == 0 else ()
        expected_outputs = ("out",) if index == len(filters) - 1 else ()
        if invocation.inputs != expected_inputs or invocation.outputs != expected_outputs:
            raise ShutterIntegrationError(
                "the execution path requires exactly [0:v] on the first filter, "
                "[out] on the last filter, and no internal links"
            )
    return filters


def _normalize_invocation(invocation: FilterInvocation) -> str:
    option_names = _SHUTTER_POSITIONAL_OPTIONS.get(invocation.name)
    if option_names is None or invocation.option_error is not None:
        return invocation.raw

    positional = invocation.positional_options()
    if not positional:
        return invocation.raw
    if len(positional) > len(option_names):
        raise ShutterIntegrationError(
            f"{invocation.name} has too many positional options to normalize"
        )

    # Shutter puts positional fields first (for example, 5200 before pl=1).
    # Refuse a novel ordering instead of assigning it an uncertain AVOption.
    saw_named = False
    ordinal = 0
    replacements: list[tuple[int, str]] = []
    for option in invocation.options:
        if option.name is not None:
            saw_named = True
            continue
        if saw_named:
            raise ShutterIntegrationError(
                f"{invocation.name} has a positional option after a named option"
            )
        relative = option.span.start - invocation.span.start
        replacements.append((relative, option_names[ordinal] + "="))
        ordinal += 1

    rendered = invocation.raw
    for relative, prefix in reversed(replacements):
        rendered = rendered[:relative] + prefix + rendered[relative:]
    return rendered


def normalize_shutter_chain(source: str) -> str:
    """Strip the two Shutter labels and normalize its positional options."""

    filters = _linear_shutter_filters(source)
    normalized = ",".join(_normalize_invocation(item) for item in filters)
    # The strict parser is the only parser allowed on the compiler path.  This
    # second parse also rejects option constructs the lenient scanner recorded
    # but could not interpret.
    try:
        parse_filtergraph(normalized)
    except FiltergraphSyntaxError as error:
        raise ShutterIntegrationError(
            f"normalized Shutter chain is outside the accepted grammar: {error}"
        ) from error
    return normalized


def _candidate_islands(filters: tuple[FilterInvocation, ...]) -> tuple[CandidateIsland, ...]:
    candidates: list[CandidateIsland] = []
    start: int | None = None
    names: list[str] = []
    formats: frozenset[str] = frozenset()

    def close(end: int) -> None:
        nonlocal start, formats
        if start is not None and end - start > 1:
            candidates.append(
                CandidateIsland(start, end, tuple(names), formats)
            )
        start = None
        names.clear()
        formats = frozenset()

    native_formats = frozenset(LAYOUTS)
    for index, invocation in enumerate(filters):
        lowerer = LOWERERS.get(invocation.name)
        if lowerer is None:
            close(index)
            continue
        try:
            lowerer(invocation, index, None)
        except LoweringError:
            close(index)
            continue

        accepted = FILTER_FORMATS[invocation.name] & native_formats
        intersection = accepted if start is None else formats & accepted
        if start is not None and not intersection:
            close(index)
            intersection = accepted
        if start is None:
            start = index
        names.append(invocation.name)
        formats = frozenset(intersection)

    close(len(filters))
    return tuple(candidates)


def _ffmpeg_version(executable: str) -> str:
    try:
        process = subprocess.run(
            [executable, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ShutterIntegrationError(f"could not query FFmpeg version: {error}") from error
    match = _VERSION.search(process.stdout)
    if process.returncode != 0 or match is None:
        raise ShutterIntegrationError("could not identify the FFmpeg version")
    return match.group(1)


def _probe_formats(
    executable: str,
    filters: tuple[FilterInvocation, ...],
    candidates: tuple[CandidateIsland, ...],
    ffmpeg_arguments: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Observe each candidate's negotiated output format in one FFmpeg run."""

    candidates_by_end = {candidate.end: index for index, candidate in enumerate(candidates)}
    probe_filters: list[str] = []
    for index, invocation in enumerate(filters, start=1):
        probe_filters.append(invocation.raw)
        probe_index = candidates_by_end.get(index)
        if probe_index is not None:
            probe_filters.append(f"showinfo@lavfi_cc_probe_{probe_index}")
    graph = "[0:v]" + ",".join(probe_filters) + "[out]"
    if ffmpeg_arguments is None:
        probe_input = [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=1",
        ]
        video_options: list[str] = []
    else:
        try:
            input_index = ffmpeg_arguments.index("-i")
        except ValueError as error:
            raise ShutterIntegrationError(
                "the export command has no separate -i input"
            ) from error
        if input_index + 1 >= len(ffmpeg_arguments):
            raise ShutterIntegrationError("the export command's -i has no input")
        # Shutter's input seek, demuxer, decoder, and hardware options precede
        # its first -i. Reuse them so the probe sees the export's real decoded
        # format, but stop before any output can be opened.
        probe_input = list(ffmpeg_arguments[: input_index + 2])
        video_options = []
        for names in (
            ("-c:v", "-codec:v", "-vcodec"),
            ("-pix_fmt", "-pixel_format"),
            ("-profile:v", "-profile"),
        ):
            selected: tuple[str, str] | None = None
            for index, item in enumerate(ffmpeg_arguments[:-1]):
                if item in names:
                    selected = (item, ffmpeg_arguments[index + 1])
            if selected is not None:
                video_options.extend(selected)

    command = [
        executable,
        "-hide_banner",
        "-nostdin",
        *probe_input,
        # An input prefix copied from a UI command can carry its own log level.
        # Put the probe level last so the tagged showinfo frames stay visible.
        "-loglevel",
        "info",
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        *video_options,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ShutterIntegrationError(
            f"FFmpeg format-negotiation probe failed: {error}"
        ) from error
    if process.returncode != 0:
        detail = process.stderr.strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise ShutterIntegrationError(
            "FFmpeg could not replay the Shutter chain for format verification: "
            + (detail or f"status {process.returncode}")
        )

    observed: dict[int, str] = {}
    for match in _PROBE_FRAME.finditer(process.stderr):
        observed.setdefault(int(match.group(1)), match.group(2).lower())
    missing = [str(index) for index in range(len(candidates)) if index not in observed]
    if missing:
        raise ShutterIntegrationError(
            "FFmpeg did not report a negotiated format for probe(s) " + ", ".join(missing)
        )
    return tuple(observed[index] for index in range(len(candidates)))


def _pin_chain(
    filters: tuple[FilterInvocation, ...], pins: tuple[FormatPin, ...]
) -> str:
    pins_by_start = {pin.start: pin for pin in pins}
    output: list[str] = []
    for index, invocation in enumerate(filters):
        pin = pins_by_start.get(index)
        if pin is not None:
            output.append(f"format={pin.pixel_format}")
        output.append(invocation.raw)
    return ",".join(output)


def prepare_shutter_graph(
    source: str,
    *,
    ffmpeg: str | os.PathLike[str],
    expected_version: str = EXPECTED_FFMPEG_VERSION,
    ffmpeg_arguments: Sequence[str] | None = None,
) -> PreparedGraph:
    """Normalize, verify, pin, and lower one Shutter filtergraph."""

    executable = str(Path(ffmpeg).resolve())
    actual_version = _ffmpeg_version(executable)
    if actual_version != expected_version:
        raise ShutterIntegrationError(
            f"expected pinned FFmpeg {expected_version}, found {actual_version}"
        )

    normalized = normalize_shutter_chain(source)
    filters = parse_filtergraph(normalized).filters
    candidates = _candidate_islands(filters)
    if not candidates:
        raise ShutterIntegrationError(
            "the Shutter chain has no compatible run of at least two supported filters"
        )
    formats = _probe_formats(
        executable,
        filters,
        candidates,
        ffmpeg_arguments=ffmpeg_arguments,
    )
    pins: list[FormatPin] = []
    for candidate, pixel_format in zip(candidates, formats, strict=True):
        if pixel_format not in LAYOUTS:
            continue
        if pixel_format not in candidate.compatible_formats:
            continue
        pins.append(
            FormatPin(
                candidate.start,
                candidate.end,
                pixel_format,
                candidate.names,
            )
        )
    if not pins:
        detail = ", ".join(
            f"{'/'.join(candidate.names)} negotiated {pixel_format}"
            for candidate, pixel_format in zip(candidates, formats, strict=True)
        )
        raise ShutterIntegrationError(
            "no candidate negotiated a common native format" + (f": {detail}" if detail else "")
        )

    pinned = _pin_chain(filters, tuple(pins))
    analysis = analyze_filtergraph(pinned, auto_islands=True)
    if not analysis.eligible:
        reason = "; ".join(item.format() for item in analysis.diagnostics)
        raise ShutterIntegrationError(
            "verified graph did not lower into a profitable island: " + reason
        )
    probe_scope = "export_command" if ffmpeg_arguments is not None else "synthetic"
    return PreparedGraph(
        source,
        normalized,
        pinned,
        tuple(pins),
        analysis,
        probe_scope,
    )


def _execute(executable: str, arguments: Sequence[str]) -> int:
    try:
        return subprocess.run([executable, *arguments], check=False).returncode
    except OSError as error:
        print(f"lavfi-cc-shutter: could not execute FFmpeg: {error}", file=sys.stderr)
        return 127


def _fallback(
    executable: str,
    arguments: Sequence[str],
    reason: str,
    *,
    require_fusion: bool,
    status: int,
) -> int:
    if require_fusion:
        print(f"lavfi-cc-shutter: fusion required: {reason}", file=sys.stderr)
        return status
    print(
        f"lavfi-cc-shutter: fusion unavailable ({reason}); running original command",
        file=sys.stderr,
    )
    return _execute(executable, arguments)


def _preflight(
    executable: str, replacement: str, pixel_format: str
) -> tuple[bool, str]:
    try:
        process = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s=2x2:r=1,format={pixel_format}",
                "-vf",
                replacement,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if process.returncode == 0:
        return True, ""
    detail = process.stderr.strip()
    return False, (detail[-2000:] if detail else f"status {process.returncode}")


def run_shutter_ffmpeg(
    arguments: Sequence[str],
    *,
    ffmpeg: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    expected_version: str = EXPECTED_FFMPEG_VERSION,
    require_fusion: bool = False,
) -> int:
    """Run a Shutter-shaped FFmpeg command through verified fusion."""

    try:
        executable = resolve_ffmpeg(ffmpeg)
    except FFmpegIntegrationError as error:
        print(f"lavfi-cc-shutter: {error}", file=sys.stderr)
        return 127
    try:
        location = find_filter_complex_argument(arguments)
        prepared = prepare_shutter_graph(
            location.value,
            ffmpeg=executable,
            expected_version=expected_version,
            ffmpeg_arguments=arguments,
        )
    except ShutterIntegrationError as error:
        return _fallback(
            executable,
            arguments,
            str(error),
            require_fusion=require_fusion,
            status=2,
        )

    try:
        cache = KernelCache(cache_dir)
        with ExitStack() as stack:
            replacements: list[str] = []
            for plan in prepared.analysis.plans:
                cached = stack.enter_context(
                    cache.acquire(plan.ir, compiler=compile_kernel)
                )
                replacement = fused_filter(
                    prepared.analysis,
                    cached.library_path,
                    cached.library_path.parent,
                    plan=plan,
                )
                replacements.append(replacement)
                available, detail = _preflight(
                    executable, replacement, plan.ir.layout
                )
                if not available:
                    return _fallback(
                        executable,
                        arguments,
                        f"fused-filter preflight failed: {detail}",
                        require_fusion=require_fusion,
                        status=1,
                    )

            rewritten_chain = rewrite_filtergraph_islands(
                prepared.analysis, replacements
            )
            rewritten_graph = f"[0:v]{rewritten_chain}[out]"
            rewritten_arguments = list(arguments)
            rewritten_arguments[location.value_index] = rewritten_graph
            return _execute(executable, rewritten_arguments)
    except (CacheError, NativeError, FFmpegIntegrationError, OSError) as error:
        return _fallback(
            executable,
            arguments,
            f"kernel compilation or validation failed: {error}",
            require_fusion=require_fusion,
            status=1,
        )


def proxy_main(argv: list[str] | None = None) -> int:
    """Act as an FFmpeg executable for a Shutter test installation.

    Non-export probes and commands without ``-filter_complex`` pass through
    silently. Set ``LAVFI_CC_FFMPEG`` to the real pinned, patched executable so
    the proxy cannot accidentally resolve itself through ``PATH``.
    """

    forwarded = list(sys.argv[1:] if argv is None else argv)
    try:
        executable = resolve_ffmpeg()
    except FFmpegIntegrationError as error:
        print(f"lavfi-cc-shutter: {error}", file=sys.stderr)
        return 127
    try:
        if Path(executable).resolve() == Path(sys.argv[0]).resolve():
            raise ShutterIntegrationError(
                "LAVFI_CC_FFMPEG resolves to the Shutter proxy itself"
            )
    except (OSError, ShutterIntegrationError) as error:
        print(f"lavfi-cc-shutter: could not resolve proxy paths: {error}", file=sys.stderr)
        return 127

    if "-filter_complex" not in forwarded:
        return _execute(executable, forwarded)
    require_fusion = os.environ.get(
        "LAVFI_CC_SHUTTER_REQUIRE_FUSION", ""
    ).lower() in {"1", "true", "yes", "on"}
    return run_shutter_ffmpeg(
        forwarded,
        ffmpeg=executable,
        expected_version=os.environ.get(
            "LAVFI_CC_SHUTTER_FFMPEG_VERSION", EXPECTED_FFMPEG_VERSION
        ),
        require_fusion=require_fusion,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lavfi-cc-shutter", description=MVP_NAME)
    commands = parser.add_subparsers(dest="command", required=True)

    explain = commands.add_parser(
        "explain", help="verify and show the rewrite for one Shutter graph"
    )
    explain.add_argument("--filter-complex", required=True, metavar="GRAPH")
    explain.add_argument("--ffmpeg", metavar="PATH")
    explain.add_argument(
        "--expected-version",
        default=os.environ.get(
            "LAVFI_CC_SHUTTER_FFMPEG_VERSION", EXPECTED_FFMPEG_VERSION
        ),
    )
    explain.add_argument("--json", action="store_true")

    run = commands.add_parser(
        "run", help="compile verified islands and run the Shutter FFmpeg command"
    )
    run.add_argument("--ffmpeg", metavar="PATH")
    run.add_argument("--cache-dir", metavar="PATH")
    run.add_argument(
        "--expected-version",
        default=os.environ.get(
            "LAVFI_CC_SHUTTER_FFMPEG_VERSION", EXPECTED_FFMPEG_VERSION
        ),
    )
    run.add_argument("--require-fusion", action="store_true")
    run.add_argument("ffmpeg_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "explain":
        try:
            executable = resolve_ffmpeg(arguments.ffmpeg)
            prepared = prepare_shutter_graph(
                arguments.filter_complex,
                ffmpeg=executable,
                expected_version=arguments.expected_version,
            )
        except (FFmpegIntegrationError, ShutterIntegrationError) as error:
            print(f"lavfi-cc-shutter: {error}", file=sys.stderr)
            return 2
        if arguments.json:
            json.dump(prepared.as_dict(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print(MVP_NAME)
            print(f"Normalized chain: {prepared.normalized_chain}")
            print(f"Format probe scope: {prepared.probe_scope}")
            for pin in prepared.pins:
                print(
                    f"Verified format: filters [{pin.start}:{pin.end}] "
                    f"{','.join(pin.filters)} -> {pin.pixel_format}"
                )
            print(f"Frame passes eliminated: {prepared.analysis.eliminated_passes}")
            print(f"Planned rewrite: {prepared.placeholder_filtergraph}")
        return 0

    forwarded = list(arguments.ffmpeg_arguments)
    if not forwarded or forwarded[0] != "--":
        print(
            "lavfi-cc-shutter: run arguments must be separated with --",
            file=sys.stderr,
        )
        return 2
    return run_shutter_ffmpeg(
        forwarded[1:],
        ffmpeg=arguments.ffmpeg,
        cache_dir=arguments.cache_dir,
        expected_version=arguments.expected_version,
        require_fusion=arguments.require_fusion,
    )


if __name__ == "__main__":
    raise SystemExit(main())
