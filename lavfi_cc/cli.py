"""Command-line interface for the frontend and reference interpreter."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import BinaryIO

from . import __version__
from .frontend import Analysis, analyze_filtergraph
from .interpreter import InterpreterError, interpret_rgba8


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
    return parser


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


def _print_analysis(analysis: Analysis) -> None:
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
    print("Compiler passes: none (reference semantics)")
    print("Reference interpreter: available (Week 3)")
    print("Cache status: unavailable until the native backend milestone")
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


def _interpret(arguments: argparse.Namespace) -> int:
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
                    interpret_rgba8(
                        analysis.ir, frame, arguments.width, arguments.height
                    )
                )
                frame_index += 1
            output_stream.flush()
    except (InterpreterError, OSError) as error:
        print(f"lavfi-cc: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "explain":
        analysis = analyze_filtergraph(arguments.vf)
        if arguments.json:
            json.dump(analysis.as_dict(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_analysis(analysis)
        return 0 if analysis.eligible else 2
    if arguments.command == "interpret":
        return _interpret(arguments)
    raise AssertionError(f"unhandled command {arguments.command}")
