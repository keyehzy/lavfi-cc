"""Deterministic, readable scalar C generation for the pixel IR.

A layout whose planes all share the frame's resolution is walked one pixel at a
time, loading every channel of that pixel so a stage may mix them.  A
chroma-subsampled layout has no such common iteration space -- one ``yuv420p``
chroma sample covers four luma samples -- so it is walked one plane at a time
at that plane's own resolution.  Only channel-independent stages can be emitted
that way, which :func:`lavfi_cc.interpreter.validate_ir` has already enforced by
the time code generation runs.

Sample width enters in exactly two places: the type of a row pointer and the
type of a table entry.  Everything between them is the same text at eight bits
and above, and the eight-bit rendering is character for character what it was
before higher depths existed, so no cached or bundled kernel was invalidated by
adding them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .expr import ExprProgram, c_helpers, render_c
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
    size = len(rows[0])
    lines = [f"static const {c_type} {name}[{len(rows)}][{size}] = {{"]
    for row in rows:
        lines.append("    {")
        for start in range(0, size, 16):
            values = ", ".join(str(value) for value in row[start : start + 16])
            lines.append(f"        {values},")
        lines.append("    },")
    lines.append("};")
    return lines


MixerColumn = tuple[tuple[int, int, int, int], ...]


def _format_vector_table(
    name: str, entries: MixerColumn, vector_type: str
) -> list[str]:
    lines = [f"static const {vector_type} {name}[{len(entries)}] = {{"]
    for start in range(0, len(entries), 4):
        values = ", ".join(
            "{" + ", ".join(str(value) for value in entry) + "}"
            for entry in entries[start : start + 4]
        )
        lines.append(f"    {values},")
    lines.append("};")
    return lines


def _mixer_columns(operation: Operation) -> tuple[MixerColumn, ...]:
    tables = operation.parameters["contribution_tables"]
    size = len(tables[0][0])
    return tuple(
        tuple(
            tuple(tables[output][input_][value] for output in range(4))
            for value in range(size)
        )
        for input_ in range(4)
    )


def _mixer_column_mode(column: MixerColumn) -> str:
    zero = (0,) * len(column)
    identity = tuple(range(len(column)))
    output_tables = tuple(
        tuple(entry[output] for entry in column) for output in range(4)
    )
    if all(table == zero for table in output_tables):
        return "zero"
    if all(table in (zero, identity) for table in output_tables):
        return "direct"
    return "table"


def _mixer_vector_type(depth: int) -> str:
    """The vector the mixer sums its four terms in.

    A term is a sample scaled by at most two, so eight-bit terms and their sum
    fit an ``int16_t``; above eight bits they do not, which is the whole reason
    the type is named after its width rather than fixed.
    """

    return "lavfi_i16x4" if depth == 8 else "lavfi_i32x4"


def _mixer_element_type(depth: int) -> str:
    return "int16_t" if depth == 8 else "int32_t"


def _direct_mixer_vector(input_: int, column: MixerColumn, depth: int) -> str:
    zero = (0,) * len(column)
    lanes = [
        "0"
        if tuple(entry[output] for entry in column) == zero
        else f"({_mixer_element_type(depth)})c{input_}"
        for output in range(4)
    ]
    return f"({_mixer_vector_type(depth)})" + "{" + ", ".join(lanes) + "}"


def _sample_index(layout: PixelLayout, channel: int) -> str:
    """Render the index of one channel's sample within its plane row.

    The index is in samples rather than bytes, which is what a row pointer of
    the layout's own sample type wants.
    """

    offset = layout.offset(channel)
    assert offset is not None
    if layout.step == 1:
        return "x" if offset == 0 else f"x + {offset}"
    scaled = f"x * {layout.step}"
    return scaled if offset == 0 else f"{scaled} + {offset}"


def _is_lut(operation: Operation) -> bool:
    return operation.kind in {"lut8", "lut16"}


def _is_mixer(operation: Operation) -> bool:
    return (
        operation.kind == "matrix4x4"
        and operation.parameters.get("evaluation") in MIXER_EVALUATIONS
    )


def _is_chroma_rotate(operation: Operation) -> bool:
    return operation.kind == "chroma_rotate_i32"


def _is_expr(operation: Operation) -> bool:
    return operation.kind == "expr_f32"


def _expr_program(operation: Operation) -> ExprProgram:
    return ExprProgram.from_dict(operation.parameters["program"])


def _stage_channels(operation: Operation) -> frozenset[int]:
    """The channels a stage writes, and therefore the loops it belongs in."""

    if _is_chroma_rotate(operation):
        return frozenset({1, 2})
    if _is_expr(operation):
        return _expr_program(operation).channels_written
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
    if _is_expr(operation):
        program = _expr_program(operation)
        return program.channels_read | program.channels_written
    return frozenset()


def _stage_declarations(stages: list[Operation], depth: int) -> list[str]:
    sample_type = _sample_type(depth)
    lines: list[str] = []
    for index, operation in enumerate(stages):
        if _is_lut(operation):
            noun = "byte" if depth == 8 else f"{depth}-bit sample"
            lines.append(
                f"/* stage {index}: four independent {noun} lookup tables */"
            )
            rows = tuple(tuple(row) for row in operation.parameters["tables"])
            lines.extend(_format_table(f"lut_{index}", sample_type, rows))
        elif _is_chroma_rotate(operation):
            lines.append(
                f"/* stage {index}: chroma rotation, evaluated inline "
                f"(cos {operation.parameters['cosine']}, "
                f"sin {operation.parameters['sine']} in 16.16) */"
            )
        elif _is_expr(operation):
            program = _expr_program(operation)
            fused = sum(
                1
                for instruction in program.instructions
                if instruction[0] == "fma"
            )
            lines.append(
                f"/* stage {index}: cross-channel float32 expression, "
                f"{len(program.instructions)} operations, {fused} fused "
                f"multiply-add{'s' if fused != 1 else ''} */"
            )
        elif operation.parameters["evaluation"] == LEVELS_EVALUATION:
            lines.append(
                f"/* stage {index}: materialized levels_f32_fma (single-rounding oracle) */"
            )
            lines.extend(
                _format_table(
                    f"levels_{index}",
                    sample_type,
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
                            f"mixer_{index}_input_{input_}",
                            column,
                            _mixer_vector_type(depth),
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


def _table_index(depth: int, expression: str) -> str:
    """Render a table subscript, clamping a sample the format cannot hold.

    A table covers the format's own domain, ``1 << depth`` entries.  At eight
    and sixteen bits that is every value the sample type can hold and the
    subscript is the sample itself; in between, a malformed frame could carry a
    value the table has no entry for, and it is clamped rather than read past
    the end.  See :mod:`lavfi_cc.interpreter` on the domain this is outside of.
    """

    if depth in (8, 16):
        return expression
    maximum = (1 << depth) - 1
    return f"{expression} > {maximum} ? {maximum} : {expression}"


def _channel_assignment(
    index: int, operation: Operation, channel: int, depth: int
) -> str:
    """Render one channel's update for a stage that reads only that channel."""

    subscript = _table_index(depth, f"c{channel}")
    if _is_lut(operation):
        return f"c{channel} = lut_{index}[{channel}][{subscript}];"
    if operation.parameters.get("evaluation") == LEVELS_EVALUATION:
        return f"c{channel} = levels_{index}[{channel}][{subscript}];"
    raise ValueError(f"unsupported code-generation stage {operation.kind!r}")


def _chroma_rotate_body(
    index: int, operation: Operation, indent: str, depth: int
) -> list[str]:
    """Emit ``vf_hue.c``'s chroma rotation inline rather than as its tables.

    Upstream indexes two tables by the ``(u, v)`` pair -- 64 KiB each at eight
    bits, 2 MiB each at ten; the arithmetic behind them is a few int32
    operations that cannot overflow at these magnitudes.  The ``>> 16`` is on a
    signed value, exactly as upstream writes it, so both are the same
    arithmetic shift under the same compiler.

    ``apply_lut10`` clamps each sample into the depth before indexing, which
    ``apply_lut`` does not need to; the clamp below is that, and it is the only
    difference between the two paths.
    """

    cosine = operation.parameters["cosine"]
    sine = operation.parameters["sine"]
    centre = 1 << (depth - 1)
    rounding = f"({1 << 15} + {centre << 16})"
    load = (
        (lambda channel: f"(int32_t)c{channel}")
        if depth == 8
        else (lambda channel: f"(int32_t)({_table_index(depth, f'c{channel}')})")
    )
    return [
        f"{indent}{{",
        f"{indent}    const int32_t u{index} = {load(1)} - {centre};",
        f"{indent}    const int32_t v{index} = {load(2)} - {centre};",
        f"{indent}    c1 = lavfi_saturate_i32("
        f"(({cosine} * u{index}) - ({sine} * v{index}) + {rounding}) >> 16);",
        f"{indent}    c2 = lavfi_saturate_i32("
        f"(({sine} * u{index}) + ({cosine} * v{index}) + {rounding}) >> 16);",
        f"{indent}}}",
    ]


def _stage_body(
    index: int,
    operation: Operation,
    depth: int,
    channels: tuple[int, ...] = (0, 1, 2, 3),
    indent: str = "            ",
) -> list[str]:
    """Emit one stage for a loop carrying exactly *channels*."""

    if _is_chroma_rotate(operation):
        return _chroma_rotate_body(index, operation, indent, depth)
    if _is_expr(operation):
        return render_c(_expr_program(operation), f"e{index}", channels, indent)
    if not _is_mixer(operation):
        return [
            indent + _channel_assignment(index, operation, channel, depth)
            for channel in channels
        ]

    vector = _mixer_vector_type(depth)
    lines = [f"{indent}{vector} n{index} = ({vector}){{0, 0, 0, 0}};"]
    for input_, column in enumerate(_mixer_columns(operation)):
        mode = _mixer_column_mode(column)
        if mode == "table":
            subscript = _table_index(depth, f"c{input_}")
            term = f"mixer_{index}_input_{input_}[{subscript}]"
        elif mode == "direct":
            term = _direct_mixer_vector(input_, column, depth)
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
    width = "" if layout.depth == 8 else f", {layout.depth}-bit samples"
    detail = (
        f"{layout.name} ({kind}, {layout.plane_count} plane"
        f"{'s' if layout.plane_count != 1 else ''}, step {layout.step}{width})"
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


def _sample_type(depth: int) -> str:
    return "uint8_t" if depth == 8 else "uint16_t"


def _row_pointer(depth: int, name: str, buffer: str, stride: str, plane: int) -> str:
    """Declare one row pointer of the layout's sample type.

    The ABI hands over byte pointers and byte strides whatever the depth is, so
    a wide layout does the pointer arithmetic in bytes and casts once, which is
    also what every upstream slice function does.
    """

    constant = "const " if buffer == "src" else ""
    address = f"{buffer}[{plane}] + (ptrdiff_t)y * {stride}[{plane}]"
    if depth == 8:
        return f"{constant}uint8_t *{name} = {address};"
    return f"{constant}uint16_t *{name} = ({constant}uint16_t *)({address});"


def _pixel_walk(layout: PixelLayout, stages: list[Operation]) -> list[str]:
    """Walk one pixel at a time, loading every channel so a stage may mix them."""

    depth = layout.depth
    sample = _sample_type(depth)
    used_planes = sorted({layout.plane(channel) for channel in layout.stored_channels})
    lines = ["    for (int y = 0; y < height; ++y) {"]
    for plane in used_planes:
        lines.append(
            "        "
            + _row_pointer(
                depth, f"source_row_{plane}", "src", "src_stride", plane  # type: ignore[arg-type]
            )
        )
        lines.append(
            "        "
            + _row_pointer(
                depth, f"destination_row_{plane}", "dst", "dst_stride", plane  # type: ignore[arg-type]
            )
        )
    lines.append("        for (int x = 0; x < width; ++x) {")
    for channel in range(4):
        plane = layout.plane(channel)
        if plane is None:
            # Upstream reads no alpha for these layouts and contributes none.
            lines.append(f"            {sample} c{channel} = 0;")
        else:
            index = _sample_index(layout, channel)
            lines.append(
                f"            {sample} c{channel} = source_row_{plane}[{index}];"
            )
    for index, operation in enumerate(stages):
        lines.append("")
        lines.append(f"            /* stage {index} */")
        lines.extend(_stage_body(index, operation, depth))
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

    depth = layout.depth
    sample_type = _sample_type(depth)
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
                "            "
                + _row_pointer(
                    depth, row("source", plane), "src", "src_stride", plane
                )
            )
            lines.append(
                "            "
                + _row_pointer(
                    depth, row("destination", plane), "dst", "dst_stride", plane
                )
            )
        lines.append("            for (int x = 0; x < plane_width; ++x) {")
        for channel in channels:
            sample = _sample_index(layout, channel)
            source = row("source", layout.plane(channel))  # type: ignore[arg-type]
            lines.append(
                f"                {sample_type} c{channel} = {source}[{sample}];"
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
                _stage_body(index, operation, depth, active, "                ")
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
    # saturate their int32 results to a sample.
    uses_mixer = any(_is_mixer(operation) for operation in stages)
    needs_saturation = uses_mixer or any(
        _is_chroma_rotate(operation) for operation in stages
    )
    uses_expression = any(_is_expr(operation) for operation in stages)
    layout = get_layout(ir.layout)
    depth = layout.depth
    sample = _sample_type(depth)
    maximum = layout.max_value

    lines = [
        "/* Generated by lavfi-cc. This file is deterministic for its input IR. */",
        f"/* source plan: {ir.plan_hash} */",
        f"/* optimized plan: {passes.ir.plan_hash} */",
        '#include "kernel_abi.h"',
        "",
    ]
    if layout.high_depth:
        # Samples are stored little-endian and loaded as native words, so this
        # translation unit is only correct where the two agree.
        lines.extend(
            [
                "#if defined(__BYTE_ORDER__) && "
                "__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__",
                f'#error "{layout.name} kernels load native 16-bit samples and '
                'need a little-endian host"',
                "#endif",
                "",
            ]
        )
    if uses_expression:
        # lrintf and fmaf. Both are exactly specified by IEEE-754, unlike the
        # pow and sin this compiler refuses to depend on.
        lines.extend(["#include <math.h>", ""])
        lines.extend([*c_helpers(depth), ""])
    if uses_mixer:
        element = _mixer_element_type(depth)
        lines.extend(
            [
                f"typedef {element} {_mixer_vector_type(depth)} "
                "__attribute__((ext_vector_type(4)));",
                "",
            ]
        )
    if needs_saturation:
        lines.extend(
            [
                f"static inline {sample} lavfi_saturate_i32(int32_t value)",
                "{",
                "    if (value < 0)",
                "        return 0;",
                f"    if (value > {maximum})",
                f"        return {maximum};",
                f"    return ({sample})value;",
                "}",
                "",
            ]
        )
    lines.extend(_stage_declarations(stages, depth))
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
