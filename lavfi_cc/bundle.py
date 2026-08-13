"""Ahead-of-time kernel bundles, so production never needs a compiler.

The persistent cache compiles a kernel the first time a plan is seen, which
means a deployment needs Clang on every host that might meet a new filtergraph.
A *bundle* moves that work to build time: the kernels for a known corpus of
filtergraphs are generated, compiled, and indexed by plan hash once, and the
resulting directory is shipped as an ordinary build artifact.

At run time a bundle is consulted before the cache.  A hit is checksum-,
ABI-, layout-, and plan-hash-validated exactly like a cache entry, so a stale
or tampered bundle is rejected rather than trusted.

``--emit-only`` writes the generated C and a manifest without compiling
anything, so a project's own build system can compile the kernels with its own
toolchain and flags.  That path needs no Clang at all, at build time or later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .codegen import generate_c
from .frontend import analyze_filtergraph
from .native import (
    BASE_COMPILER_FLAGS,
    NativeError,
    NativeKernel,
    compile_generated_c,
    library_suffix,
)
from .version import VERSION


BUNDLE_SCHEMA_VERSION = 1
INDEX_NAME = "index.json"


class BundleError(RuntimeError):
    """A bundle could not be built, read, or trusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BundleEntry:
    plan_hash: str
    layout: str
    source_name: str
    library_name: str | None
    library_sha256: str | None
    canonical_filters: tuple[str, ...]
    graphs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "layout": self.layout,
            "source": self.source_name,
            "library": self.library_name,
            "library_sha256": self.library_sha256,
            "canonical_filters": list(self.canonical_filters),
            "graphs": list(self.graphs),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "BundleEntry":
        if not isinstance(value, dict):
            raise BundleError("bundle entry is not an object")
        try:
            return cls(
                str(value["plan_hash"]),
                str(value["layout"]),
                str(value["source"]),
                value["library"] if value["library"] is None else str(value["library"]),
                value["library_sha256"]
                if value["library_sha256"] is None
                else str(value["library_sha256"]),
                tuple(str(item) for item in value.get("canonical_filters", ())),
                tuple(str(item) for item in value.get("graphs", ())),
            )
        except KeyError as error:
            raise BundleError(f"bundle entry is missing {error.args[0]!r}") from error


@dataclass
class BuildReport:
    entries: list[BundleEntry] = field(default_factory=list)
    graphs: int = 0
    ineligible: list[tuple[str, str]] = field(default_factory=list)

    @property
    def compiled(self) -> int:
        return sum(1 for entry in self.entries if entry.library_name)


def build_bundle(
    graphs: Sequence[str],
    output_directory: str | os.PathLike[str],
    *,
    auto_islands: bool = False,
    compile_kernels: bool = True,
    clang: str | os.PathLike[str] | None = None,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> BuildReport:
    """Generate, optionally compile, and index every kernel a corpus needs."""

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    report = BuildReport()
    by_hash: dict[str, BundleEntry] = {}
    sources: dict[str, list[str]] = {}

    for graph in graphs:
        report.graphs += 1
        analysis = analyze_filtergraph(graph, auto_islands=auto_islands)
        if not analysis.eligible:
            reason = "; ".join(item.format() for item in analysis.diagnostics)
            report.ineligible.append((graph, reason))
            continue
        for plan in analysis.plans:
            sources.setdefault(plan.ir.plan_hash, []).append(graph)
            if plan.ir.plan_hash in by_hash:
                continue
            generated = generate_c(
                plan.ir,
                identity_elimination=identity_elimination,
                lut_composition=lut_composition,
            )
            source_name = f"{plan.ir.plan_hash}.c"
            source_path = directory / source_name
            source_path.write_text(generated.source, encoding="utf-8")

            library_name: str | None = None
            library_sha: str | None = None
            if compile_kernels:
                library_name = plan.ir.plan_hash + library_suffix()
                try:
                    compile_generated_c(
                        generated,
                        directory / library_name,
                        source_path=source_path,
                        clang=clang,
                    )
                except NativeError as error:
                    raise BundleError(
                        f"could not compile kernel {plan.ir.plan_hash}: {error}"
                    ) from error
                library_sha = _sha256_file(directory / library_name)

            by_hash[plan.ir.plan_hash] = BundleEntry(
                plan.ir.plan_hash,
                plan.ir.layout,
                source_name,
                library_name,
                library_sha,
                plan.canonical_filters,
                (),
            )

    report.entries = [
        BundleEntry(
            entry.plan_hash,
            entry.layout,
            entry.source_name,
            entry.library_name,
            entry.library_sha256,
            entry.canonical_filters,
            tuple(dict.fromkeys(sources.get(entry.plan_hash, ()))),
        )
        for entry in by_hash.values()
    ]
    report.entries.sort(key=lambda entry: entry.plan_hash)

    index = {
        "schema": BUNDLE_SCHEMA_VERSION,
        "compiler_version": VERSION,
        "compiler_flags": list(BASE_COMPILER_FLAGS),
        "compiled": compile_kernels,
        "entries": [entry.as_dict() for entry in report.entries],
    }
    (directory / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


class Bundle:
    """A prebuilt, validated set of kernels indexed by plan hash."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.root = Path(directory).expanduser().resolve()
        index_path = self.root / INDEX_NAME
        if not index_path.is_file():
            raise BundleError(f"bundle index does not exist: {index_path}")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BundleError(f"bundle index is unreadable: {error}") from error
        if not isinstance(index, dict):
            raise BundleError("bundle index is not an object")
        if index.get("schema") != BUNDLE_SCHEMA_VERSION:
            raise BundleError(
                f"bundle schema {index.get('schema')!r} is not "
                f"{BUNDLE_SCHEMA_VERSION}"
            )
        entries = index.get("entries")
        if not isinstance(entries, list):
            raise BundleError("bundle index has no entries")
        self.entries = {
            entry.plan_hash: entry
            for entry in (BundleEntry.from_dict(value) for value in entries)
        }

    def __contains__(self, plan_hash: object) -> bool:
        return plan_hash in self.entries

    def library_path(self, plan_hash: str) -> Path:
        """Return the validated library for *plan_hash*."""

        entry = self.entries.get(plan_hash)
        if entry is None:
            raise BundleError(f"bundle has no kernel for plan {plan_hash}")
        if not entry.library_name:
            raise BundleError(
                f"bundle entry {plan_hash} was emitted without a compiled library"
            )
        path = self.root / entry.library_name
        if not path.is_file():
            raise BundleError(f"bundle kernel does not exist: {path}")
        if entry.library_sha256 and _sha256_file(path) != entry.library_sha256:
            raise BundleError(f"bundle kernel checksum does not match: {path}")
        return path

    def open_kernel(self, plan_hash: str, layout: str) -> NativeKernel:
        """Load a bundled kernel, checking its ABI, layout, and plan hash."""

        entry = self.entries[plan_hash]
        if entry.layout != layout:
            raise BundleError(
                f"bundle kernel {plan_hash} is {entry.layout!r}, not {layout!r}"
            )
        try:
            return NativeKernel(self.library_path(plan_hash), plan_hash, layout=layout)
        except NativeError as error:
            raise BundleError(f"bundled kernel failed validation: {error}") from error


def resolve_bundle(
    configured: str | os.PathLike[str] | None = None,
) -> Bundle | None:
    """Open the configured bundle, or the one named by ``LAVFI_CC_BUNDLE``."""

    requested = os.fspath(configured) if configured is not None else None
    requested = requested or os.environ.get("LAVFI_CC_BUNDLE")
    if not requested:
        return None
    return Bundle(requested)


def load_graph_sources(paths: Iterable[str]) -> list[str]:
    from .scanner import load_graphs

    graphs: list[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            graphs.extend(load_graphs(stream))
    return graphs
