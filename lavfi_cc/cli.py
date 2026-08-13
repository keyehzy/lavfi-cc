"""Command-line interface for the frontend and reference interpreter."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shlex
import sys
from typing import BinaryIO, Callable

from . import __version__
from .frontend import Analysis, analyze_filtergraph
from .interpreter import InterpreterError, interpret_rgba8
from .ir import PixelIR
from .native import (
    NativeError,
    NativeKernel,
    compile_kernel,
    library_suffix,
)
from .passes import PassResult, optimize_ir


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


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
    interpret = subparsers.add_parser(
        "interpret", help="run an eligible chain on packed raw RGBA8 frames"
    )
    interpret.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    interpret.add_argument("--width", required=True, type=_positive_integer)
    interpret.add_argument("--height", required=True, type=_positive_integer)
    interpret.add_argument(
        "--input",
        default="-",
        metavar="PATH",
        help="packed RGBA8 input (default: standard input)",
    )
    interpret.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="packed RGBA8 output (default: standard output)",
    )
    native = subparsers.add_parser(
        "native", help="compile and run an eligible chain on raw RGBA8 frames"
    )
    native.add_argument("--vf", required=True, metavar="FILTERGRAPH")
    native.add_argument("--width", required=True, type=_positive_integer)
    native.add_argument("--height", required=True, type=_positive_integer)
    native.add_argument(
        "--input",
        default="-",
        metavar="PATH",
        help="packed RGBA8 input (default: standard input)",
    )
    native.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="packed RGBA8 output (default: standard output)",
    )
    _add_pass_switches(native)

    compile_parser = subparsers.add_parser(
        "compile", help="generate and compile a standalone native kernel"
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
        help="library path (default: .build/week4/<plan-hash>.<platform>)",
    )
    compile_parser.add_argument(
        "--emit-c",
        metavar="PATH",
        help="write generated C to this path instead of beside the library",
    )
    _add_pass_switches(compile_parser)
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
        help="disable adjacent LUT composition",
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


def _print_analysis(analysis: Analysis, passes: PassResult | None = None) -> None:
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

    assert analysis.ir is not None
    assert analysis.region is not None
    start, end = analysis.region
    print(f"Eligibility: eligible (filters [{start}:{end}])")
    print("Canonical region:")
    for filter_ in analysis.canonical_filters:
        print(f"  {filter_}")
    print("IR:")
    print(analysis.ir.pretty())
    assert passes is not None
    print("Compiler passes:")
    for name, changes in passes.changes:
        print(f"  {name}: {changes} change{'s' if changes != 1 else ''}")
    print(f"Optimized plan hash: {passes.ir.plan_hash}")
    print("Reference interpreter: available (Week 3)")
    print("Native C backend: available (Week 4; uncached)")
    print("Cache status: unavailable until the Week 6 cache milestone")
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
    analysis = analyze_filtergraph(arguments.vf)
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None

    if arguments.input != "-" and arguments.output != "-":
        if Path(arguments.input).resolve() == Path(arguments.output).resolve():
            print("lavfi-cc: input and output paths must be different", file=sys.stderr)
            return 2

    frame_size = arguments.width * arguments.height * 4
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
    analysis = analyze_filtergraph(arguments.vf)
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None
    try:
        with NativeKernel.compile(
            analysis.ir,
            identity_elimination=not arguments.no_identity_elimination,
            lut_composition=not arguments.no_lut_composition,
        ) as kernel:
            return _stream_frames(
                arguments,
                lambda _ir, source, width, height: kernel.process_rgba8(
                    source, width, height
                ),
            )
    except NativeError as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1


def _compile(arguments: argparse.Namespace) -> int:
    analysis = analyze_filtergraph(arguments.vf)
    if not analysis.eligible:
        for diagnostic in analysis.diagnostics:
            print(f"lavfi-cc: {diagnostic.format()}", file=sys.stderr)
        return 2
    assert analysis.ir is not None
    try:
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
    except (NativeError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1
    print(f"Plan hash: {artifact.generated.plan_hash}")
    print(f"Optimized plan hash: {artifact.generated.optimized_plan_hash}")
    for name, changes in artifact.generated.passes.changes:
        print(f"Pass {name}: {changes} change{'s' if changes != 1 else ''}")
    print(f"Generated C: {artifact.source_path}")
    print(f"Native library: {artifact.library_path}")
    print(f"Compiler command: {shlex.join(artifact.command)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "explain":
        analysis = analyze_filtergraph(arguments.vf)
        passes = optimize_ir(analysis.ir) if analysis.ir is not None else None
        if arguments.json:
            value = analysis.as_dict()
            value["optimization"] = passes.as_dict() if passes is not None else None
            value["native_backend"] = "available_uncached" if passes is not None else None
            json.dump(value, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_analysis(analysis, passes)
        return 0 if analysis.eligible else 2
    if arguments.command == "interpret":
        return _interpret(arguments)
    if arguments.command == "native":
        return _native(arguments)
    if arguments.command == "compile":
        return _compile(arguments)
    raise AssertionError(f"unhandled command {arguments.command}")
