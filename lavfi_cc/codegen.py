"""Deterministic, readable scalar C generation for the RGBA8 pixel IR.

A layout whose planes all share the frame's resolution is walked one pixel at a
time, loading every channel of that pixel so a stage may mix them.  A
chroma-subsampled layout has no such common iteration space -- one ``yuv420p``
chroma sample covers four luma samples -- so it is walked one plane at a time
at that plane's own resolution.  Only channel-independent stages can be emitted
that way, which :func:`lavfi_cc.interpreter.validate_ir` has already enforced by
the time code generation runs.
"""

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


def _is_chroma_rotate(operation: Operation) -> bool:
    return operation.kind == "chroma_rotate_i32"


def _stage_channels(operation: Operation) -> frozenset[int]:
    """The channels a stage writes, and therefore the loops it belongs in."""

    if _is_chroma_rotate(operation):
        return frozenset({1, 2})
    return frozenset(range(4))


def _stage_coupling(operation: Operation) -> frozenset[int]:
    """The channels a stage must see together in one loop iteration.

    Empty when each output depends only on its own input, which is the case for
    every table-driven stage: those write all four channels but couple none of
    them, so their planes stay in separate loops.
    """

    if _is_chroma_rotate(operation):
        return frozenset({1, 2})
    if _is_mixer(operation):
        return frozenset(range(4))
    return frozenset()


def _stage_declarations(stages: list[Operation]) -> list[str]:
    lines: list[str] = []
    for index, operation in enumerate(stages):
        if operation.kind == "lut8":
            lines.append(f"/* stage {index}: four independent byte lookup tables */")
            rows = tuple(tuple(row) for row in operation.parameters["tables"])
            lines.extend(_format_table(f"lut_{index}", "uint8_t", rows))
        elif _is_chroma_rotate(operation):
            lines.append(
                f"/* stage {index}: chroma rotation, evaluated inline "
                f"(cos {operation.parameters['cosine']}, "
                f"sin {operation.parameters['sine']} in 16.16) */"
            )
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


def _channel_assignment(index: int, operation: Operation, channel: int) -> str:
    """Render one channel's update for a stage that reads only that channel."""

    if operation.kind == "lut8":
        return f"c{channel} = lut_{index}[{channel}][c{channel}];"
    if operation.parameters.get("evaluation") == LEVELS_EVALUATION:
        return f"c{channel} = levels_{index}[{channel}][c{channel}];"
    raise ValueError(f"unsupported code-generation stage {operation.kind!r}")


def _chroma_rotate_body(index: int, operation: Operation, indent: str) -> list[str]:
    """Emit ``vf_hue.c``'s chroma rotation inline rather than as its tables.

    Upstream indexes two 64 KiB tables by the ``(u, v)`` pair; the arithmetic
    behind them is a few int32 operations that cannot overflow at these
    magnitudes.  The ``>> 16`` is on a signed value, exactly as upstream writes
    it, so both are the same arithmetic shift under the same compiler.
    """

    cosine = operation.parameters["cosine"]
    sine = operation.parameters["sine"]
    rounding = f"({1 << 15} + {128 << 16})"
    return [
        f"{indent}{{",
        f"{indent}    const int32_t u{index} = (int32_t)c1 - 128;",
        f"{indent}    const int32_t v{index} = (int32_t)c2 - 128;",
        f"{indent}    c1 = lavfi_saturate_i32("
        f"(({cosine} * u{index}) - ({sine} * v{index}) + {rounding}) >> 16);",
        f"{indent}    c2 = lavfi_saturate_i32("
        f"(({sine} * u{index}) + ({cosine} * v{index}) + {rounding}) >> 16);",
        f"{indent}}}",
    ]


def _stage_body(
    index: int,
    operation: Operation,
    channels: tuple[int, ...] = (0, 1, 2, 3),
    indent: str = "            ",
) -> list[str]:
    """Emit one stage for a loop carrying exactly *channels*."""

    if _is_chroma_rotate(operation):
        return _chroma_rotate_body(index, operation, indent)
    if not _is_mixer(operation):
        return [
            indent + _channel_assignment(index, operation, channel)
            for channel in channels
        ]

    lines = [f"{indent}lavfi_i16x4 n{index} = (lavfi_i16x4){{0, 0, 0, 0}};"]
    for input_, column in enumerate(_mixer_columns(operation)):
        mode = _mixer_column_mode(column)
        if mode == "table":
            term = f"mixer_{index}_input_{input_}[c{input_}]"
        elif mode == "direct":
            term = _direct_mixer_vector(input_, column)
        else:
            continue
        lines.append(f"{indent}n{index} += {term};")
    lines.extend(
        f"{indent}c{channel} = lavfi_saturate_i32(n{index}[{channel}]);"
        for channel in channels
    )
    return lines


def _layout_comment(layout: PixelLayout) -> str:
    kind = "planar" if layout.planar else "packed"
    detail = (
        f"{layout.name} ({kind}, {layout.plane_count} plane"
        f"{'s' if layout.plane_count != 1 else ''}, step {layout.step})"
    )
    if layout.subsampled:
        shifts = ", ".join(
            f"plane {plane} is {1 << layout.plane_shift(plane)[0]}x"
            f"{1 << layout.plane_shift(plane)[1]} subsampled"
            for plane in range(layout.plane_count)
            if layout.plane_shift(plane) != (0, 0)
        )
        detail += f"; {shifts}"
    return f"/* byte layout: {detail} */"


def _pixel_walk(layout: PixelLayout, stages: list[Operation]) -> list[str]:
    """Walk one pixel at a time, loading every channel so a stage may mix them."""

    used_planes = sorted({layout.plane(channel) for channel in layout.stored_channels})
    lines = ["    for (int y = 0; y < height; ++y) {"]
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
        lines.append(f"            destination_row_{plane}[{index}] = c{channel};")
    lines.extend(["        }", "    }"])
    return lines


def _ceil_shift_expression(name: str, shift: int) -> str:
    """Render FFmpeg's AV_CEIL_RSHIFT, which is how it sizes a chroma plane."""

    if shift == 0:
        return name
    return f"({name} + {(1 << shift) - 1}) >> {shift}"


def _walk_planes(
    layout: PixelLayout, stages: list[Operation]
) -> list[tuple[int, ...]]:
    """Group the planes that one loop has to visit together.

    The default is one loop per plane, which is what every channel-independent
    pipeline needs.  A stage that reads across channels -- ``hue`` rotating Cb
    into Cr -- forces the planes holding those channels into a single loop, so
    both samples are in hand at once.  ``validate_ir`` has already refused any
    stage that would join planes the layout samples differently.
    """

    groups = [(plane,) for plane, channels in enumerate(layout.plane_channels) if channels]
    for operation in stages:
        wanted = {
            layout.plane(channel)
            for channel in _stage_coupling(operation)
            if layout.plane(channel) is not None
        }
        if len(wanted) < 2:
            continue
        merged = tuple(
            sorted(
                plane
                for group in groups
                if wanted & set(group)
                for plane in group
            )
        )
        groups = [group for group in groups if not (wanted & set(group))]
        groups.append(merged)
        groups.sort()
    return groups


def _plane_walk(layout: PixelLayout, stages: list[Operation]) -> list[str]:
    """Walk each group of co-sited planes at its own resolution.

    ``width`` and ``height`` describe plane 0, so every other plane derives its
    dimensions the way the rest of libavfilter does.  A group holding one plane
    -- the case for every channel-independent pipeline -- names its row
    pointers plainly; a group spanning planes qualifies them by plane number.
    """

    lines: list[str] = []
    for group in _walk_planes(layout, stages):
        channels = tuple(
            channel for plane in group for channel in layout.plane_channels[plane]
        )
        if not channels:
            continue
        # Every plane of a group shares one resolution, by construction.
        horizontal, vertical = layout.plane_shift(group[0])
        stored = [channel for channel in channels if channel in layout.stored_channels]

        def row(prefix: str, plane: int) -> str:
            return f"{prefix}_row" if len(group) == 1 else f"{prefix}_row_{plane}"

        lines.append(
            f"    /* plane{'s' if len(group) != 1 else ''} "
            f"{', '.join(str(plane) for plane in group)} carr"
            f"{'y' if len(group) != 1 else 'ies'} channel"
            f"{'s' if len(channels) != 1 else ''} "
            f"{', '.join(str(channel) for channel in channels)} */"
        )
        lines.append("    {")
        lines.append(
            f"        const int plane_width = {_ceil_shift_expression('width', horizontal)};"
        )
        lines.append(
            f"        const int plane_height = {_ceil_shift_expression('height', vertical)};"
        )
        lines.append("        for (int y = 0; y < plane_height; ++y) {")
        for plane in group:
            lines.append(
                f"            const uint8_t *{row('source', plane)} = "
                f"src[{plane}] + (ptrdiff_t)y * src_stride[{plane}];"
            )
            lines.append(
                f"            uint8_t *{row('destination', plane)} = "
                f"dst[{plane}] + (ptrdiff_t)y * dst_stride[{plane}];"
            )
        lines.append("            for (int x = 0; x < plane_width; ++x) {")
        for channel in channels:
            sample = _sample_index(layout, channel)
            source = row("source", layout.plane(channel))  # type: ignore[arg-type]
            lines.append(
                f"                uint8_t c{channel} = {source}[{sample}];"
            )
        for index, operation in enumerate(stages):
            active = tuple(
                channel for channel in channels
                if channel in _stage_channels(operation)
            )
            if not active:
                continue
            lines.append("")
            lines.append(f"                /* stage {index} */")
            lines.extend(
                _stage_body(index, operation, active, "                ")
            )
        lines.append("")
        for channel in stored:
            sample = _sample_index(layout, channel)
            destination = row("destination", layout.plane(channel))  # type: ignore[arg-type]
            lines.append(f"                {destination}[{sample}] = c{channel};")
        lines.extend(["            }", "        }", "    }"])
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
    # The vector type is only for the mixer; both it and the chroma rotation
    # saturate their int32 results to bytes.
    uses_mixer = any(_is_mixer(operation) for operation in stages)
    needs_saturation = uses_mixer or any(
        _is_chroma_rotate(operation) for operation in stages
    )

    lines = [
        "/* Generated by lavfi-cc. This file is deterministic for its input IR. */",
        f"/* source plan: {ir.plan_hash} */",
        f"/* optimized plan: {passes.ir.plan_hash} */",
        '#include "kernel_abi.h"',
        "",
    ]
    if uses_mixer:
        lines.extend(
            ["typedef int16_t lavfi_i16x4 __attribute__((ext_vector_type(4)));", ""]
        )
    if needs_saturation:
        lines.extend(
            [
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
    lines.extend(_stage_declarations(stages))
    lines.extend(
        [
            _layout_comment(layout),
            "static void process_rgba8(",
            "    uint8_t *const *dst, const ptrdiff_t *dst_stride,",
            "    const uint8_t *const *src, const ptrdiff_t *src_stride,",
            "    int width, int height)",
            "{",
        ]
    )
    if layout.subsampled:
        lines.extend(_plane_walk(layout, stages))
    else:
        lines.extend(_pixel_walk(layout, stages))
    lines.extend(
        [
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
