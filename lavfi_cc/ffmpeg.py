"""Week 5 integration with the patched FFmpeg ``fused`` AVFilter."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

from .frontend import Analysis, analyze_filtergraph
from .native import NativeError, NativeKernel, compile_kernel, library_suffix


class FFmpegIntegrationError(RuntimeError):
    """The wrapper cannot safely identify or rewrite one video filtergraph."""


@dataclass(frozen=True)
class FiltergraphArgument:
    option_index: int
    value_index: int
    value: str


def find_filtergraph_argument(arguments: Sequence[str]) -> FiltergraphArgument:
    """Find the one narrow ``-vf``/``-filter:v`` form accepted by the MVP."""

    matches: list[FiltergraphArgument] = []
    for index, argument in enumerate(arguments):
        if argument not in {"-vf", "-filter:v"}:
            continue
        if index + 1 >= len(arguments):
            raise FFmpegIntegrationError(f"{argument} is missing its filtergraph")
        matches.append(FiltergraphArgument(index, index + 1, arguments[index + 1]))
    if not matches:
        raise FFmpegIntegrationError(
            "no -vf or -filter:v filtergraph was found in the FFmpeg arguments"
        )
    if len(matches) != 1:
        raise FFmpegIntegrationError(
            "multiple -vf/-filter:v arguments are ambiguous; exactly one is required"
        )
    return matches[0]


def _quote_filter_value(value: str) -> str:
    """Quote one AVOption string for FFmpeg's filtergraph parser."""

    if "\0" in value or "\n" in value or "\r" in value:
        raise FFmpegIntegrationError("kernel paths may not contain control characters")
    # Outside quotes, a backslash makes the following quote literal. Reopening
    # the quoted section after it handles the only character special in quotes.
    escaped = value.replace("\\", "\\\\").replace("'", "'\\''")
    return "'" + escaped + "'"


def fused_filter(
    analysis: Analysis,
    kernel_path: str | os.PathLike[str],
    kernel_root: str | os.PathLike[str],
) -> str:
    """Render the checked dynamic-filter invocation for an eligible analysis."""

    if not analysis.eligible or analysis.ir is None:
        raise FFmpegIntegrationError("cannot render a fused filter for an ineligible graph")
    remove_color = int(
        "remove_color_dependent_side_data" in analysis.ir.metadata_effects
    )
    return (
        "fused="
        f"kernel={_quote_filter_value(str(Path(kernel_path).resolve()))}:"
        f"kernel_root={_quote_filter_value(str(Path(kernel_root).resolve()))}:"
        f"plan_hash={analysis.ir.plan_hash}:"
        f"remove_color_side_data={remove_color}"
    )


def rewrite_filtergraph(analysis: Analysis, replacement: str) -> str:
    """Replace only the frontend-approved region, retaining both format guards."""

    if not analysis.eligible or analysis.graph is None or analysis.region is None:
        raise FFmpegIntegrationError("cannot rewrite an ineligible filtergraph")
    start, end = analysis.region
    filters = [invocation.raw for invocation in analysis.graph.filters]
    return ",".join(filters[:start] + [replacement] + filters[end:])


def rewrite_ffmpeg_arguments(
    arguments: Sequence[str], location: FiltergraphArgument, filtergraph: str
) -> list[str]:
    rewritten = list(arguments)
    rewritten[location.value_index] = filtergraph
    return rewritten


def _default_ffmpeg_candidates() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    host = {"Darwin": "macos", "Linux": "linux"}.get(platform.system())
    if host is None:
        return ()
    return (root / ".build" / f"ffmpeg-week5-{host}" / "bin" / "ffmpeg",)


def resolve_ffmpeg(configured: str | os.PathLike[str] | None = None) -> str:
    requested = os.fspath(configured) if configured is not None else None
    requested = requested or os.environ.get("LAVFI_CC_FFMPEG")
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.parent == Path("."):
            found = shutil.which(requested)
            if found:
                candidate = Path(found)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise FFmpegIntegrationError(f"FFmpeg is not executable: {candidate}")
    for candidate in _default_ffmpeg_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).resolve())
    raise FFmpegIntegrationError(
        "FFmpeg was not found; pass --ffmpeg or set LAVFI_CC_FFMPEG"
    )


def _execute(ffmpeg: str, arguments: Sequence[str]) -> int:
    try:
        return subprocess.run([ffmpeg, *arguments], check=False).returncode
    except OSError as error:
        print(f"lavfi-cc: could not execute FFmpeg: {error}", file=sys.stderr)
        return 127


def _preflight(ffmpeg: str, replacement: str) -> tuple[bool, str]:
    """Load and execute the kernel before running the user's real command."""

    try:
        process = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=2x2:r=1,format=rgba",
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
        )
    except OSError as error:
        return False, str(error)
    if process.returncode == 0:
        return True, ""
    detail = process.stderr.strip()
    if len(detail) > 2000:
        detail = detail[-2000:]
    return False, detail or f"FFmpeg exited with status {process.returncode}"


def _fallback(
    ffmpeg: str,
    arguments: Sequence[str],
    reason: str,
    *,
    require_fusion: bool,
    failure_status: int,
) -> int:
    if require_fusion:
        print(f"lavfi-cc: fusion required: {reason}", file=sys.stderr)
        return failure_status
    print(
        f"lavfi-cc: fusion unavailable ({reason}); running original chain",
        file=sys.stderr,
    )
    return _execute(ffmpeg, arguments)


def run_ffmpeg(
    arguments: Sequence[str],
    *,
    ffmpeg: str | os.PathLike[str] | None = None,
    require_fusion: bool = False,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> int:
    """Compile, preflight, rewrite, and run one ordinary FFmpeg invocation."""

    try:
        executable = resolve_ffmpeg(ffmpeg)
    except FFmpegIntegrationError as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 127

    try:
        location = find_filtergraph_argument(arguments)
    except FFmpegIntegrationError as error:
        return _fallback(
            executable,
            arguments,
            str(error),
            require_fusion=require_fusion,
            failure_status=2,
        )

    analysis = analyze_filtergraph(location.value)
    if not analysis.eligible:
        reason = "; ".join(item.format() for item in analysis.diagnostics)
        return _fallback(
            executable,
            arguments,
            reason,
            require_fusion=require_fusion,
            failure_status=2,
        )
    assert analysis.ir is not None

    build_root = Path(__file__).resolve().parents[1] / ".build" / "week5"
    try:
        build_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(build_root, 0o700)
        with tempfile.TemporaryDirectory(prefix="run-", dir=build_root) as temporary:
            directory = Path(temporary).resolve()
            os.chmod(directory, 0o700)
            library = directory / (analysis.ir.plan_hash + library_suffix())
            source = directory / (analysis.ir.plan_hash + ".c")
            compile_kernel(
                analysis.ir,
                library,
                source_path=source,
                identity_elimination=identity_elimination,
                lut_composition=lut_composition,
            )
            os.chmod(library, 0o600)
            os.chmod(source, 0o600)
            # Apply the same ABI/hash checks as the AVFilter before involving
            # the user's command. This also catches a corrupt compiler result.
            with NativeKernel(library, analysis.ir.plan_hash):
                pass
            replacement = fused_filter(analysis, library, directory)
            rewritten_graph = rewrite_filtergraph(analysis, replacement)
            rewritten_arguments = rewrite_ffmpeg_arguments(
                arguments, location, rewritten_graph
            )
            available, detail = _preflight(executable, replacement)
            if not available:
                return _fallback(
                    executable,
                    arguments,
                    f"fused-filter preflight failed: {detail}",
                    require_fusion=require_fusion,
                    failure_status=1,
                )
            return _execute(executable, rewritten_arguments)
    except (NativeError, FFmpegIntegrationError, OSError) as error:
        return _fallback(
            executable,
            arguments,
            f"kernel compilation or validation failed: {error}",
            require_fusion=require_fusion,
            failure_status=1,
        )
