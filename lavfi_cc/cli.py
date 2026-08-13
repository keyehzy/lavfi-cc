"""Command-line interface for the frontend and reference interpreter."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shlex
import sys
from typing import BinaryIO, Callable

from . import __version__
from .bundle import BundleError, build_bundle, load_graph_sources
from .cache import CacheError, KernelCache
from .frontend import Analysis, IslandPlan, analyze_filtergraph
from .ffmpeg import run_ffmpeg
from .interpreter import InterpreterError, interpret_rgba8
from .ir import PixelIR
from .layouts import get_layout
from .native import (
    NativeError,
    NativeKernel,
    compile_kernel,
    library_suffix,
)
from .passes import PassResult, optimize_ir
from .scanner import load_graphs, render_text, scan_graph, summarize


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _byte_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGT]i?B|B)?", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected bytes or a size such as 512MiB or 1GiB"
        )
    number = int(match.group(1))
    suffix = (match.group(2) or "B").upper()
    powers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    return number * powers[suffix]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lavfi-cc")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    explain = subparsers.add_parser(
        "explain", help="parse a -vf chain and explain fusion eligibility"
    )
    explain.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    explain.add_argument(
        "--json", action="store_true", help="emit diagnostics and IR as JSON"
    )
    explain.add_argument("--cache-dir", metavar="PATH", help="kernel cache directory")
    _add_pass_switches(explain)
    interpret = subparsers.add_parser(
        "interpret", help="run an eligible chain on raw frames in the chain's own format"
    )
    interpret.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    interpret.add_argument("--width", required=True, type=_positive_integer)
    interpret.add_argument("--height", required=True, type=_positive_integer)
    interpret.add_argument(
        "--input",
        default="-",
        metavar="PATH",
        help="raw input in the layout the chain pins (default: standard input)",
    )
    interpret.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="raw output in the layout the chain pins (default: standard output)",
    )
    interpret.add_argument(
        "--auto-islands",
        action="store_true",
        help="discover fusible islands anywhere instead of requiring explicit "
        "format boundaries",
    )
    native = subparsers.add_parser(
        "native", help="compile and run an eligible chain on raw frames in the chain's own format"
    )
    native.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    native.add_argument("--width", required=True, type=_positive_integer)
    native.add_argument("--height", required=True, type=_positive_integer)
    native.add_argument(
        "--input",
        default="-",
        metavar="PATH",
        help="raw input in the layout the chain pins (default: standard input)",
    )
    native.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="raw output in the layout the chain pins (default: standard output)",
    )
    native.add_argument("--cache-dir", metavar="PATH", help="kernel cache directory")
    _add_pass_switches(native)

    compile_parser = subparsers.add_parser(
        "compile", help="ensure an eligible native kernel is cached"
    )
    compile_parser.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    compile_parser.add_argument(
        "--pixel-format",
        choices=("rgba", "rgba8"),
        default="rgba",
        help="internal pixel format (the Week 4 backend supports RGBA8)",
    )
    compile_parser.add_argument(
        "--output",
        metavar="PATH",
        help="export library path (supplying this bypasses the cache)",
    )
    compile_parser.add_argument(
        "--emit-c",
        metavar="PATH",
        help="export generated C (supplying this bypasses the cache)",
    )
    compile_parser.add_argument(
        "--cache-dir", metavar="PATH", help="kernel cache directory"
    )
    _add_pass_switches(compile_parser)
    run = subparsers.add_parser(
        "run", help="compile an eligible -vf region and run patched FFmpeg"
    )
    run.add_argument(
        "--ffmpeg", metavar="PATH", help="FFmpeg executable (or LAVFI_CC_FFMPEG)"
    )
    run.add_argument(
        "--require-fusion",
        action="store_true",
        help="fail instead of running the original chain when fusion is unavailable",
    )
    run.add_argument("--cache-dir", metavar="PATH", help="kernel cache directory")
    run.add_argument(
        "--bundle",
        metavar="DIR",
        help="prebuilt kernel bundle to prefer over compiling (or LAVFI_CC_BUNDLE)",
    )
    run.add_argument(
        "--require-bundle",
        action="store_true",
        help="never invoke a compiler; every kernel must come from the bundle",
    )
    _add_pass_switches(run)
    run.add_argument(
        "ffmpeg_arguments",
        nargs=argparse.REMAINDER,
        help="FFmpeg arguments, preceded by --",
    )
    bundle = subparsers.add_parser(
        "bundle",
        help="build kernels ahead of time so deployment needs no compiler",
    )
    bundle_source = bundle.add_mutually_exclusive_group(required=True)
    bundle_source.add_argument("--vf", metavar="FILTERGRAPH", help="one filtergraph")
    bundle_source.add_argument(
        "--file",
        metavar="PATH",
        action="append",
        help="a corpus with one filtergraph per line (repeatable)",
    )
    bundle.add_argument(
        "--output", required=True, metavar="DIR", help="bundle directory to write"
    )
    bundle.add_argument(
        "--emit-only",
        action="store_true",
        help="write generated C and the index without compiling anything",
    )
    bundle.add_argument("--json", action="store_true", help="emit the report as JSON")
    _add_pass_switches(bundle)

    scan = subparsers.add_parser(
        "scan",
        help="report fusible islands in arbitrary filtergraphs without compiling",
    )
    scan_source = scan.add_mutually_exclusive_group(required=True)
    scan_source.add_argument("--vf", metavar="FILTERGRAPH", help="one filtergraph")
    scan_source.add_argument(
        "--file",
        metavar="PATH",
        help="a corpus with one filtergraph per line ('-' for standard input)",
    )
    scan.add_argument(
        "--entry-format",
        metavar="PIXFMT",
        help="pixel format entering each chain, when the caller can prove it",
    )
    scan.add_argument(
        "--verbose", action="store_true", help="report every island, not only totals"
    )
    scan.add_argument("--json", action="store_true", help="emit the report as JSON")

    cache = subparsers.add_parser("cache", help="inspect or prune native-kernel cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_list = cache_commands.add_parser("list", help="list cached kernels")
    cache_list.add_argument(
        "--cache-dir", metavar="PATH", help="kernel cache directory"
    )
    cache_list.add_argument("--json", action="store_true", help="emit entries as JSON")
    cache_prune = cache_commands.add_parser("prune", help="remove oldest kernels")
    cache_prune.add_argument(
        "--cache-dir", metavar="PATH", help="kernel cache directory"
    )
    cache_prune.add_argument("--max-size", required=True, type=_byte_size, metavar="SIZE")
    return parser


def _add_pass_switches(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-identity-elimination",
        action="store_true",
        help="disable identity-stage removal",
    )
    parser.add_argument(
        "--no-lut-composition",
        action="store_true",
        help="disable levels materialization, LUT composition, and mixer folding",
    )
    parser.add_argument(
        "--auto-islands",
        action="store_true",
        help="discover fusible islands anywhere instead of requiring explicit "
        "format boundaries",
    )


def _analyze(arguments: argparse.Namespace, *, entry_format: str | None = None) -> Analysis:
    """Analyze ``--vf`` honouring the island-discovery switches of any command."""

    return analyze_filtergraph(
        arguments.vf,
        auto_islands=getattr(arguments, "auto_islands", False),
        entry_format=entry_format,
    )


def _format_filter(index: int, invocation: object) -> str:
    rendered_options = []
    for option in invocation.options:
        if option.name is None:
            rendered_options.append(repr(option.value))
        else:
            rendered_options.append(f"{option.name}={option.value!r}")
    suffix = " " + " ".join(rendered_options) if rendered_options else ""
    return (
        f"  [{index}] {invocation.name}{suffix} "
        f"(bytes {invocation.span.start}:{invocation.span.end})"
    )


def _print_island(
    analysis: Analysis,
    plan: IslandPlan,
    passes: PassResult | None,
    cache: dict[str, object] | None,
) -> None:
    start, end = plan.region
    detail = f" via {plan.boundary} boundary" if analysis.auto_islands else ""
    print(f"Eligibility: eligible (filters [{start}:{end}]{detail})")
    print("Canonical region:")
    for filter_ in plan.canonical_filters:
        print(f"  {filter_}")
    print("IR:")
    print(plan.ir.pretty())
    assert passes is not None
    print("Compiler passes:")
    for name, changes in passes.changes:
        print(f"  {name}: {changes} change{'s' if changes != 1 else ''}")
    print(f"Optimized plan hash: {passes.ir.plan_hash}")
    if cache is None:
        print("Cache status: unavailable")
    else:
        print(f"Cache key: {cache['key']}")
        print(f"Cache status: {cache['status']}")
        if cache.get("path"):
            print(f"Cached kernel: {cache['path']}")
        if cache.get("detail"):
            print(f"Cache detail: {cache['detail']}")


def _print_analysis(
    analysis: Analysis,
    details: list[tuple[IslandPlan, PassResult | None, dict[str, object] | None]],
) -> None:
    print("Parsed filters:")
    if analysis.graph is None:
        print("  <filtergraph did not parse>")
    else:
        for index, invocation in enumerate(analysis.graph.filters):
            print(_format_filter(index, invocation))

    if not analysis.eligible:
        print("Eligibility: unsupported")
        for diagnostic in analysis.diagnostics:
            print(f"  - {diagnostic.format()}")
        return

    for position, (plan, passes, cache) in enumerate(details):
        if len(details) > 1:
            print(f"Island {position + 1} of {len(details)}:")
        _print_island(analysis, plan, passes, cache)
    print(f"Frame passes eliminated: {analysis.eliminated_passes}")
    print("Reference interpreter: available (Week 3)")
    print("Native C backend: available (Week 4)")
    print("FFmpeg fused-filter integration: available (Week 5)")
    print(f"Planned rewrite: {analysis.rewritten_filtergraph}")


def _read_frame(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_frames(
    arguments: argparse.Namespace,
    process_frame: Callable[[PixelIR, bytes, int, int], bytes],
) -> int:
    analysis = _analyze(arguments, entry_format="rgba")
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None

    if arguments.input != "-" and arguments.output != "-":
        if Path(arguments.input).resolve() == Path(arguments.output).resolve():
            print("lavfi-cc: input and output paths must be different", file=sys.stderr)
            return 2

    # The chain names the layout, and a planar or subsampled one is not
    # width * height * 4 bytes.
    frame_size = get_layout(analysis.ir.layout).frame_size(
        arguments.width, arguments.height
    )
    try:
        with ExitStack() as stack:
            input_stream = (
                sys.stdin.buffer
                if arguments.input == "-"
                else stack.enter_context(open(arguments.input, "rb"))
            )
            output_stream = (
                sys.stdout.buffer
                if arguments.output == "-"
                else stack.enter_context(open(arguments.output, "wb"))
            )
            frame_index = 0
            while True:
                frame = _read_frame(input_stream, frame_size)
                if not frame:
                    break
                if len(frame) != frame_size:
                    raise InterpreterError(
                        f"partial frame {frame_index}: read {len(frame)} bytes, "
                        f"expected {frame_size}"
                    )
                output_stream.write(
                    process_frame(
                        analysis.ir, frame, arguments.width, arguments.height
                    )
                )
                frame_index += 1
            output_stream.flush()
    except (InterpreterError, NativeError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1
    return 0


def _interpret(arguments: argparse.Namespace) -> int:
    return _stream_frames(arguments, interpret_rgba8)


def _native(arguments: argparse.Namespace) -> int:
    analysis = _analyze(arguments, entry_format="rgba")
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None
    try:
        cache = KernelCache(arguments.cache_dir)
        with cache.acquire(
            analysis.ir,
            identity_elimination=not arguments.no_identity_elimination,
            lut_composition=not arguments.no_lut_composition,
        ) as cached, NativeKernel(
            cached.library_path, analysis.ir.plan_hash, layout=analysis.ir.layout
        ) as kernel:
            return _stream_frames(
                arguments,
                lambda _ir, source, width, height: kernel.process_rgba8(
                    source, width, height
                ),
            )
    except (CacheError, NativeError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1


def _compile(arguments: argparse.Namespace) -> int:
    analysis = _analyze(arguments, entry_format="rgba")
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None
    try:
        if arguments.output or arguments.emit_c:
            output = (
                Path(arguments.output)
                if arguments.output
                else Path(".build")
                / "week4"
                / (analysis.ir.plan_hash + library_suffix())
            )
            artifact = compile_kernel(
                analysis.ir,
                output,
                source_path=arguments.emit_c,
                identity_elimination=not arguments.no_identity_elimination,
                lut_composition=not arguments.no_lut_composition,
            )
            generated = artifact.generated
            print(f"Plan hash: {generated.plan_hash}")
            print(f"Optimized plan hash: {generated.optimized_plan_hash}")
            for name, changes in generated.passes.changes:
                print(f"Pass {name}: {changes} change{'s' if changes != 1 else ''}")
            print("Cache status: bypassed (explicit output)")
            print(f"Generated C: {artifact.source_path}")
            print(f"Native library: {artifact.library_path}")
            print(f"Compiler command: {shlex.join(artifact.command)}")
            return 0

        cache = KernelCache(arguments.cache_dir)
        cached = cache.ensure(
            analysis.ir,
            identity_elimination=not arguments.no_identity_elimination,
            lut_composition=not arguments.no_lut_composition,
        )
    except (CacheError, NativeError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1
    print(f"Plan hash: {cached.generated.plan_hash}")
    print(f"Optimized plan hash: {cached.generated.optimized_plan_hash}")
    for name, changes in cached.generated.passes.changes:
        print(f"Pass {name}: {changes} change{'s' if changes != 1 else ''}")
    print(f"Cache key: {cached.key}")
    print(f"Cache status: {cached.status}")
    print(f"Generated C: {cached.source_path}")
    print(f"Native library: {cached.library_path}")
    if cached.command is not None:
        print(f"Compiler command: {shlex.join(cached.command)}")
    return 0


def _bundle(arguments: argparse.Namespace) -> int:
    if arguments.vf is not None:
        graphs = [arguments.vf]
    else:
        try:
            graphs = load_graph_sources(arguments.file)
        except OSError as error:
            print(f"lavfi-cc: {error}", file=sys.stderr)
            return 1
    if not graphs:
        print("lavfi-cc: no filtergraph to bundle", file=sys.stderr)
        return 2

    try:
        report = build_bundle(
            graphs,
            arguments.output,
            auto_islands=arguments.auto_islands,
            compile_kernels=not arguments.emit_only,
            identity_elimination=not arguments.no_identity_elimination,
            lut_composition=not arguments.no_lut_composition,
        )
    except (BundleError, NativeError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        json.dump(
            {
                "directory": str(Path(arguments.output).resolve()),
                "graphs": report.graphs,
                "kernels": len(report.entries),
                "compiled": report.compiled,
                "entries": [entry.as_dict() for entry in report.entries],
                "ineligible": [
                    {"graph": graph, "reason": reason}
                    for graph, reason in report.ineligible
                ],
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(f"Bundle directory: {Path(arguments.output).resolve()}")
        print(f"Graphs read: {report.graphs}")
        print(f"Distinct kernels: {len(report.entries)}")
        print(f"Compiled libraries: {report.compiled}")
        for entry in report.entries:
            print(f"  {entry.plan_hash} {entry.layout} {entry.source_name}")
        if report.ineligible:
            print(f"Graphs with no fusible island: {len(report.ineligible)}")
    # A corpus is allowed to contain graphs with nothing to fuse.
    return 0


def _scan(arguments: argparse.Namespace) -> int:
    if arguments.vf is not None:
        graphs = [arguments.vf]
    else:
        try:
            if arguments.file == "-":
                graphs = load_graphs(sys.stdin)
            else:
                with open(arguments.file, "r", encoding="utf-8") as stream:
                    graphs = load_graphs(stream)
        except OSError as error:
            print(f"lavfi-cc: {error}", file=sys.stderr)
            return 1
    if not graphs:
        print("lavfi-cc: no filtergraph to scan", file=sys.stderr)
        return 2

    scans = [
        scan_graph(graph, entry_format=arguments.entry_format) for graph in graphs
    ]
    summary = summarize(scans)
    if arguments.json:
        json.dump(
            {
                "summary": summary.as_dict(),
                "graphs": [scan.as_dict() for scan in scans],
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        print(
            render_text(
                scans, summary, verbose=arguments.verbose or len(scans) == 1
            )
        )
    # Scanning is analysis only: a graph with no island is a finding, not a failure.
    return 0 if summary.parsed == summary.graphs else 1


def _cache(arguments: argparse.Namespace) -> int:
    cache = KernelCache(arguments.cache_dir)
    try:
        if arguments.cache_command == "list":
            entries = cache.list_entries()
            if arguments.json:
                json.dump(
                    {
                        "cache_directory": str(cache.root),
                        "entries": [entry.as_dict() for entry in entries],
                        "total_size": sum(entry.size for entry in entries),
                    },
                    sys.stdout,
                    indent=2,
                    sort_keys=True,
                )
                sys.stdout.write("\n")
            elif not entries:
                print(f"Cache directory: {cache.root}")
                print("Cache is empty")
            else:
                print(f"Cache directory: {cache.root}")
                for entry in entries:
                    print(
                        f"{entry.key} {entry.status} {entry.size} bytes "
                        f"plan={entry.plan_hash or '-'}"
                    )
            return 0
        if arguments.cache_command == "prune":
            result = cache.prune(arguments.max_size)
            print(f"Cache directory: {cache.root}")
            print(f"Size: {result.before_size} -> {result.after_size} bytes")
            print(f"Removed entries: {result.removed_entries}")
            if result.skipped_locked:
                print(f"Skipped active entries: {result.skipped_locked}")
            return 0 if result.after_size <= arguments.max_size else 1
    except (CacheError, NativeError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled cache command {arguments.cache_command}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "explain":
        analysis = _analyze(arguments)
        details: list[
            tuple[IslandPlan, PassResult | None, dict[str, object] | None]
        ] = []
        for plan in analysis.plans:
            passes = optimize_ir(
                plan.ir,
                identity_elimination=not arguments.no_identity_elimination,
                lut_composition=not arguments.no_lut_composition,
            )
            try:
                cache_info: dict[str, object] | None = KernelCache(
                    arguments.cache_dir
                ).probe(
                    plan.ir,
                    identity_elimination=not arguments.no_identity_elimination,
                    lut_composition=not arguments.no_lut_composition,
                )
            except (CacheError, NativeError, OSError) as error:
                cache_info = {
                    "key": "unavailable",
                    "status": "unavailable",
                    "detail": str(error),
                }
            details.append((plan, passes, cache_info))
        if arguments.json:
            value = analysis.as_dict()
            first = details[0] if details else (None, None, None)
            value["optimization"] = (
                first[1].as_dict() if first[1] is not None else None
            )
            value["native_backend"] = "available" if details else None
            value["ffmpeg_integration"] = "available" if details else None
            value["cache"] = first[2]
            for island, (_plan, island_passes, island_cache) in zip(
                value["islands"], details, strict=True
            ):
                island["optimization"] = (
                    island_passes.as_dict() if island_passes is not None else None
                )
                island["cache"] = island_cache
            json.dump(value, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_analysis(analysis, details)
        return 0 if analysis.eligible else 2
    if arguments.command == "interpret":
        return _interpret(arguments)
    if arguments.command == "native":
        return _native(arguments)
    if arguments.command == "compile":
        return _compile(arguments)
    if arguments.command == "bundle":
        return _bundle(arguments)
    if arguments.command == "scan":
        return _scan(arguments)
    if arguments.command == "cache":
        return _cache(arguments)
    if arguments.command == "run":
        forwarded = list(arguments.ffmpeg_arguments)
        if not forwarded or forwarded[0] != "--":
            print(
                "lavfi-cc: run arguments must be separated with --",
                file=sys.stderr,
            )
            return 2
        return run_ffmpeg(
            forwarded[1:],
            ffmpeg=arguments.ffmpeg,
            cache_dir=arguments.cache_dir,
            require_fusion=arguments.require_fusion,
            identity_elimination=not arguments.no_identity_elimination,
            lut_composition=not arguments.no_lut_composition,
            auto_islands=arguments.auto_islands,
            bundle_dir=arguments.bundle,
            require_bundle=arguments.require_bundle,
        )
    raise AssertionError(f"unhandled command {arguments.command}")
