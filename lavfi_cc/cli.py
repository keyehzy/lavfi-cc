"""Command-line interface for the Week 2 frontend."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .frontend import Analysis, analyze_filtergraph


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
    print("Compiler passes: none (Week 2 lowering only)")
    print("Cache status: unavailable until the native backend milestone")
    print(f"Planned rewrite: {analysis.rewritten_filtergraph}")


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
    raise AssertionError(f"unhandled command {arguments.command}")
