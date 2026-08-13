"""Typed, deterministic straight-line RGBA8 pixel IR."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .parser import SourceSpan


#: Version 3 added the byte layout, so a kernel built for one packed order is
#: never reused for another.
IR_VERSION = 3
CHANNELS = ("r", "g", "b", "a")


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
    #: Byte layout the kernel loads from and stores to. The operations
    #: themselves are layout-independent: they always see four logical
    #: channels, whether the layout names them red, green, blue, and alpha or
    #: luma, Cb, and Cr. A subsampled layout additionally restricts the
    #: operations to channel-independent ones, since its planes have no common
    #: iteration space.
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
        lines = [f"pixel_ir v{self.ir_version} {self.pixel_format} layout={self.layout}"]
        for index, operation in enumerate(self.operations):
            source = ""
            if operation.source is not None:
                source = (
                    f"  # filter[{operation.source.filter_index}] "
                    f"{operation.source.filter_name} bytes "
                    f"{operation.source.span.start}:{operation.source.span.end}"
                )
            if operation.kind == "lut8":
                tables = operation.parameters["tables"]
                summaries = []
                for channel, table in zip(CHANNELS, tables, strict=True):
                    digest = hashlib.sha256(bytes(table)).hexdigest()[:12]
                    summaries.append(
                        f"{channel}=sha256:{digest}[{table[0]},{table[64]},{table[128]},{table[255]}]"
                    )
                detail = " ".join(summaries)
            elif operation.kind == "matrix4x4":
                evaluation = operation.parameters["evaluation"]
                if evaluation == "sum_i32_lut_terms":
                    detail = "evaluation=sum_i32_lut_terms tables=4x4x256"
                else:
                    detail = (
                        f"evaluation={evaluation} "
                        f"coefficients={operation.parameters['coefficients']} "
                        f"offsets={operation.parameters['offsets']}"
                    )
            elif operation.kind == "quantize_rgba8":
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
