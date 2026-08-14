"""Typed, deterministic straight-line four-channel pixel IR.

The IR describes four logical channels of one sample depth.  Depth eight is
spelled exactly as it always was -- ``load_rgba8``, ``lut8``,
``quantize_rgba8``, pixel format ``rgba8`` -- so every plan hash minted before
higher depths existed is unchanged.  A deeper program uses the parallel
``load_rgba16``, ``lut16``, and ``quantize_rgba16`` operations and names its
depth in the pixel format, and the operations that were already depth-agnostic
(``matrix4x4``, ``chroma_rotate_i32``) carry a ``depth`` parameter that is
absent at eight bits for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .parser import SourceSpan


#: Version 3 added the byte layout, so a kernel built for one packed order is
#: never reused for another.  Higher sample depths needed no bump: they add
#: operations rather than changing what an existing one means, and a consumer
#: that does not know them refuses them as unsupported transforms.
IR_VERSION = 3
CHANNELS = ("r", "g", "b", "a")


def pixel_format_for(depth: int) -> str:
    """The ``pixel_format`` field naming four channels of *depth* bits."""

    return f"rgba{depth}"


def load_operation(depth: int) -> str:
    return "load_rgba8" if depth == 8 else "load_rgba16"


def store_operation(depth: int) -> str:
    return "store_rgba8" if depth == 8 else "store_rgba16"


def quantize_operation(depth: int) -> str:
    return "quantize_rgba8" if depth == 8 else "quantize_rgba16"


def lut_operation(depth: int) -> str:
    return "lut8" if depth == 8 else "lut16"


@dataclass(frozen=True)
class SourceRef:
    filter_index: int
    filter_name: str
    span: SourceSpan
    option_spans: tuple[tuple[str, SourceSpan], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "filter_index": self.filter_index,
            "filter_name": self.filter_name,
            "span": self.span.as_dict(),
            "option_spans": {
                name: span.as_dict() for name, span in self.option_spans
            },
        }


@dataclass(frozen=True)
class Operation:
    kind: str
    parameters: dict[str, Any]
    source: SourceRef | None = None

    def canonical_dict(self) -> dict[str, Any]:
        return {"op": self.kind, **self.parameters}

    def debug_dict(self) -> dict[str, Any]:
        value = self.canonical_dict()
        if self.source is not None:
            value = {**value, "source": self.source.as_dict()}
        return value


@dataclass(frozen=True)
class PixelIR:
    operations: tuple[Operation, ...]
    metadata_effects: tuple[str, ...] = ()
    ir_version: int = IR_VERSION
    pixel_format: str = "rgba8"
    #: Sample layout the kernel loads from and stores to. The operations
    #: themselves are layout-independent: they always see four logical
    #: channels, whether the layout names them red, green, blue, and alpha or
    #: luma, Cb, and Cr. On a subsampled layout an operation may read across
    #: only channels that share a sampling grid.
    layout: str = "rgba"

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "pixel_format": self.pixel_format,
            "layout": self.layout,
            "metadata_effects": list(self.metadata_effects),
            "operations": [operation.canonical_dict() for operation in self.operations],
        }

    def debug_dict(self) -> dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "pixel_format": self.pixel_format,
            "layout": self.layout,
            "metadata_effects": list(self.metadata_effects),
            "operations": [operation.debug_dict() for operation in self.operations],
        }

    def serialize(self) -> bytes:
        """Return the stable cache-key serialization (source locations excluded)."""

        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    @property
    def plan_hash(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()

    def pretty(self) -> str:
        # Only the display names change with the layout; the operations
        # themselves are layout-independent and so is the plan hash.
        from .layouts import LAYOUTS

        layout = LAYOUTS.get(self.layout)
        names = (
            tuple(
                name or channel
                for name, channel in zip(layout.component_names, CHANNELS)
            )
            if layout is not None
            else CHANNELS
        )
        lines = [f"pixel_ir v{self.ir_version} {self.pixel_format} layout={self.layout}"]
        for index, operation in enumerate(self.operations):
            source = ""
            if operation.source is not None:
                source = (
                    f"  # filter[{operation.source.filter_index}] "
                    f"{operation.source.filter_name} bytes "
                    f"{operation.source.span.start}:{operation.source.span.end}"
                )
            if operation.kind in {"lut8", "lut16"}:
                tables = operation.parameters["tables"]
                width = 1 if operation.kind == "lut8" else 2
                summaries = []
                for channel, table in zip(names, tables, strict=True):
                    entries = b"".join(
                        value.to_bytes(width, "little") for value in table
                    )
                    digest = hashlib.sha256(entries).hexdigest()[:12]
                    size = len(table)
                    probes = ",".join(
                        str(table[index])
                        for index in (0, size // 4, size // 2, size - 1)
                    )
                    summaries.append(f"{channel}=sha256:{digest}[{probes}]")
                detail = " ".join(summaries)
                if operation.kind == "lut16":
                    detail = f"depth={operation.parameters['depth']} " + detail
            elif operation.kind == "matrix4x4":
                evaluation = operation.parameters["evaluation"]
                if evaluation == "sum_i32_lut_terms":
                    entries = len(operation.parameters["contribution_tables"][0][0])
                    detail = f"evaluation=sum_i32_lut_terms tables=4x4x{entries}"
                else:
                    detail = (
                        f"evaluation={evaluation} "
                        f"coefficients={operation.parameters['coefficients']} "
                        f"offsets={operation.parameters['offsets']}"
                    )
            elif operation.kind == "chroma_rotate_i32":
                detail = (
                    f"cos={operation.parameters['cosine']} "
                    f"sin={operation.parameters['sine']} (16.16)"
                )
                if "depth" in operation.parameters:
                    detail += f" depth={operation.parameters['depth']}"
            elif operation.kind == "expr_f32":
                program = operation.parameters["program"]
                stored = [
                    names[channel] or CHANNELS[channel]
                    for channel, output in enumerate(program["outputs"])
                    if output is not None
                ]
                fused = sum(
                    1 for instruction in program["instructions"] if instruction[0] == "fma"
                )
                detail = (
                    f"{len(program['instructions'])} float32 ops "
                    f"({fused} fused multiply-add{'s' if fused != 1 else ''}) "
                    f"-> {'+'.join(stored) if stored else 'nothing'}"
                )
            elif operation.kind in {"quantize_rgba8", "quantize_rgba16"}:
                detail = f"mode={operation.parameters['mode']}"
            else:
                detail = ""
            lines.append(f"  {index:02d} {operation.kind} {detail}{source}".rstrip())
        if self.metadata_effects:
            lines.append("  metadata: " + ", ".join(self.metadata_effects))
        lines.append(f"  plan_hash: {self.plan_hash}")
        return "\n".join(lines)


def source_ref(filter_index: int, invocation: Any) -> SourceRef:
    return SourceRef(
        filter_index=filter_index,
        filter_name=invocation.name,
        span=invocation.span,
        option_spans=tuple(
            (option.name, option.span)
            for option in invocation.options
            if option.name is not None
        ),
    )
