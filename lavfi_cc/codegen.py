"""Deterministic, readable scalar C generation for the RGBA8 pixel IR."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Operation, PixelIR
from .layouts import PixelLayout, get_layout
from .passes import (
    LEVELS_EVALUATION,
    MIXER_EVALUATIONS,
    PassResult,
    materialize_levels_tables,
    optimize_ir,
)


@dataclass(frozen=True)
class GeneratedC:
    source: str
    plan_hash: str
    optimized_plan_hash: str
    passes: PassResult
    layout: str = "rgba"


def _format_table(
    name: str, c_type: str, rows: tuple[tuple[int, ...], ...]
) -> list[str]:
    lines = [f"static const {c_type} {name}[{len(rows)}][256] = {{"]
    for row in rows:
        lines.append("    {")
        for start in range(0, 256, 16):
            values = ", ".join(str(value) for value in row[start : start + 16])
            lines.append(f"        {values},")
        lines.append("    },")
    lines.append("};")
    return lines


MixerColumn = tuple[tuple[int, int, int, int], ...]


def _format_vector_table(name: str, entries: MixerColumn) -> list[str]:
    lines = [f"static const lavfi_i16x4 {name}[256] = {{"]
    for start in range(0, 256, 4):
        values = ", ".join(
            "{" + ", ".join(str(value) for value in entry) + "}"
            for entry in entries[start : start + 4]
        )
        lines.append(f"    {values},")
    lines.append("};")
    return lines


def _mixer_columns(operation: Operation) -> tuple[MixerColumn, ...]:
    tables = operation.parameters["contribution_tables"]
    return tuple(
        tuple(
            tuple(tables[output][input_][value] for output in range(4))
            for value in range(256)
        )
        for input_ in range(4)
    )


def _mixer_column_mode(column: MixerColumn) -> str:
    zero = (0,) * 256
    identity = tuple(range(256))
    output_tables = tuple(
        tuple(entry[output] for entry in column) for output in range(4)
    )
    if all(table == zero for table in output_tables):
        return "zero"
    if all(table in (zero, identity) for table in output_tables):
        return "direct"
    return "table"


def _direct_mixer_vector(input_: int, column: MixerColumn) -> str:
    zero = (0,) * 256
    lanes = [
        "0"
        if tuple(entry[output] for entry in column) == zero
        else f"(int16_t)c{input_}"
        for output in range(4)
    ]
    return "(lavfi_i16x4){" + ", ".join(lanes) + "}"


def _sample_index(layout: PixelLayout, channel: int) -> str:
    """Render the index of one channel's byte within its plane row."""

    offset = layout.offset(channel)
    assert offset is not None
    if layout.step == 1:
        return "x" if offset == 0 else f"x + {offset}"
    scaled = f"x * {layout.step}"
    return scaled if offset == 0 else f"{scaled} + {offset}"


def _is_mixer(operation: Operation) -> bool:
    return (
        operation.kind == "matrix4x4"
        and operation.parameters.get("evaluation") in MIXER_EVALUATIONS
    )


def _stage_declarations(stages: list[Operation]) -> list[str]:
    lines: list[str] = []
    for index, operation in enumerate(stages):
        if operation.kind == "lut8":
            lines.append(f"/* stage {index}: four independent byte lookup tables */")
            rows = tuple(tuple(row) for row in operation.parameters["tables"])
            lines.extend(_format_table(f"lut_{index}", "uint8_t", rows))
        elif operation.parameters["evaluation"] == LEVELS_EVALUATION:
            lines.append(
                f"/* stage {index}: materialized levels_f32_fma (single-rounding oracle) */"
            )
            lines.extend(
                _format_table(
                    f"levels_{index}",
                    "uint8_t",
                    materialize_levels_tables(operation),
                )
            )
        elif _is_mixer(operation):
            lines.append(
                f"/* stage {index}: packed independently rounded mixer terms */"
            )
            for input_, column in enumerate(_mixer_columns(operation)):
                mode = _mixer_column_mode(column)
                if mode == "table":
                    lines.extend(
                        _format_vector_table(
                            f"mixer_{index}_input_{input_}", column
                        )
                    )
                else:
                    lines.append(
                        f"/* input {input_}: {mode}; no contribution table */"
                    )
        else:
            raise ValueError(f"unsupported code-generation stage {operation.kind!r}")
        lines.append("")
    return lines


def _stage_body(index: int, operation: Operation) -> list[str]:
    if operation.kind == "lut8":
        return [
            f"            c{channel} = lut_{index}[{channel}][c{channel}];"
            for channel in range(4)
        ]
    if operation.parameters["evaluation"] == LEVELS_EVALUATION:
        return [
            f"            c{channel} = levels_{index}[{channel}][c{channel}];"
            for channel in range(4)
        ]

    if not _is_mixer(operation):
        raise ValueError(f"unsupported code-generation stage {operation.kind!r}")

    lines = [
        f"            lavfi_i16x4 n{index} = (lavfi_i16x4){{0, 0, 0, 0}};"
    ]
    for input_, column in enumerate(_mixer_columns(operation)):
        mode = _mixer_column_mode(column)
        if mode == "table":
            term = f"mixer_{index}_input_{input_}[c{input_}]"
        elif mode == "direct":
            term = _direct_mixer_vector(input_, column)
        else:
            continue
        lines.append(f"            n{index} += {term};")
    lines.extend(
        f"            c{channel} = lavfi_saturate_i32(n{index}[{channel}]);"
        for channel in range(4)
    )
    return lines


def generate_c(
    ir: PixelIR,
    *,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> GeneratedC:
    """Generate one C translation unit exporting the versioned kernel ABI."""

    passes = optimize_ir(
        ir,
        identity_elimination=identity_elimination,
        lut_composition=lut_composition,
    )
    body = passes.ir.operations[1:-1]
    stages = [body[index] for index in range(0, len(body), 2)]
    uses_mixer = any(_is_mixer(operation) for operation in stages)

    lines = [
        "/* Generated by lavfi-cc. This file is deterministic for its input IR. */",
        f"/* source plan: {ir.plan_hash} */",
        f"/* optimized plan: {passes.ir.plan_hash} */",
        '#include "kernel_abi.h"',
        "",
    ]
    if uses_mixer:
        lines.extend(
            [
                "typedef int16_t lavfi_i16x4 __attribute__((ext_vector_type(4)));",
                "",
                "static inline uint8_t lavfi_saturate_i32(int32_t value)",
                "{",
                "    if (value < 0)",
                "        return 0;",
                "    if (value > 255)",
                "        return 255;",
                "    return (uint8_t)value;",
                "}",
                "",
            ]
        )
    layout = get_layout(ir.layout)
    used_planes = sorted({layout.plane(channel) for channel in layout.stored_channels})
    lines.extend(_stage_declarations(stages))
    kind = "planar" if layout.planar else "packed"
    lines.extend(
        [
            f"/* byte layout: {layout.name} ({kind}, {layout.plane_count} plane"
            f"{'s' if layout.plane_count != 1 else ''}, step {layout.step}) */",
            "static void process_rgba8(",
            "    uint8_t *const *dst, const ptrdiff_t *dst_stride,",
            "    const uint8_t *const *src, const ptrdiff_t *src_stride,",
            "    int width, int height)",
            "{",
            "    for (int y = 0; y < height; ++y) {",
        ]
    )
    for plane in used_planes:
        lines.append(
            f"        const uint8_t *source_row_{plane} = "
            f"src[{plane}] + (ptrdiff_t)y * src_stride[{plane}];"
        )
        lines.append(
            f"        uint8_t *destination_row_{plane} = "
            f"dst[{plane}] + (ptrdiff_t)y * dst_stride[{plane}];"
        )
    lines.append("        for (int x = 0; x < width; ++x) {")
    for channel in range(4):
        plane = layout.plane(channel)
        if plane is None:
            # Upstream reads no alpha for these layouts and contributes none.
            lines.append(f"            uint8_t c{channel} = 0;")
        else:
            index = _sample_index(layout, channel)
            lines.append(
                f"            uint8_t c{channel} = source_row_{plane}[{index}];"
            )
    for index, operation in enumerate(stages):
        lines.append("")
        lines.append(f"            /* stage {index} */")
        lines.extend(_stage_body(index, operation))
    lines.append("")
    for channel in layout.stored_channels:
        plane = layout.plane(channel)
        index = _sample_index(layout, channel)
        lines.append(
            f"            destination_row_{plane}[{index}] = c{channel};"
        )
    lines.extend(
        [
            "        }",
            "    }",
            "}",
            "",
            f'static const char kernel_plan_hash[] = "{ir.plan_hash}";',
            "",
            "LAVFI_KERNEL_EXPORT const LavfiCompiledKernel lavfi_compiled_kernel = {",
            "    LAVFI_KERNEL_ABI_VERSION,",
            f"    {layout.abi_macro},",
            "    kernel_plan_hash,",
            "    process_rgba8,",
            "};",
            "",
        ]
    )
    return GeneratedC(
        "\n".join(lines),
        ir.plan_hash,
        passes.ir.plan_hash,
        passes,
        layout.name,
    )
