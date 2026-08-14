"""Deliberately simple scalar interpreter for the pixel IR.

**The domain a kernel is defined over.** A sample of a format with depth ``d``
means a value in ``[0, 2^d - 1]``, and that is the domain this interpreter and
the generated kernels are bit-exact over.  Outside it the accepted filters do
not agree with each other about what happens, so no compiler could match all of
them: ``vf_lut.c`` carries a full 65536-entry table and answers, ``vf_hue.c``
clamps the sample first, and ``vf_curves.c`` and ``vf_colorchannelmixer.c``
index a ``1 << d``-entry table with the raw sample and read past its end.  Every
table here is therefore sized to the format's own domain and indexed through a
clamp, which is defined and safe for a malformed frame without pretending to
reproduce an upstream answer that does not exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import struct
from typing import Any

from .expr import ExprError, ExprProgram
from .ir import (
    IR_VERSION,
    Operation,
    PixelIR,
    load_operation,
    lut_operation,
    pixel_format_for,
    quantize_operation,
    store_operation,
)
from .layouts import LAYOUTS, PixelLayout, get_layout


class InterpreterError(ValueError):
    """The IR or frame layout cannot be interpreted safely."""


Pixel = tuple[int, int, int, int]

#: Quantization modes each depth's ``quantize`` operation accepts.  The names
#: below eight bits' ``lookup_u8`` and ``saturate_i32_to_u8`` say the width they
#: produce; ``truncate_toward_zero_then_saturate`` clamps to whatever the
#: operation's own depth is, so it needs no suffix.
_SUPPORTED_QUANTIZERS = {
    8: {
        "lookup_u8",
        "truncate_toward_zero_then_saturate",
        "saturate_i32_to_u8",
        # An expression program carries a quantizer per channel, because its
        # channels do not have to agree on one.
        "expression_outputs",
    },
    16: {
        "lookup_u16",
        "truncate_toward_zero_then_saturate",
        "saturate_i32_to_u16",
        "expression_outputs",
    },
}


def _f32(value: float) -> float:
    """Round one operation result to IEEE-754 binary32."""

    return struct.unpack("=f", struct.pack("=f", value))[0]


def _saturate(value: int, maximum: int) -> int:
    return max(0, min(maximum, value))


def _sequence(value: Any, length: int, description: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InterpreterError(f"{description} must contain {length} entries")
    return tuple(value)


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InterpreterError(f"{description} must be an integer")
    return value


#: Each channel depends only on itself.
INDEPENDENT_GROUPS = ((0,), (1,), (2,), (3,))
#: Every output channel reads every input channel.
WHOLE_PIXEL_GROUPS = ((0, 1, 2, 3),)
#: The two chroma channels depend on each other but on nothing else.
CHROMA_PAIR_GROUPS = ((0,), (1, 2), (3,))


def _lookup(table: tuple[int, ...], value: int) -> int:
    """Index a table sized to the format's domain, clamping a stray sample.

    A well-formed sample is always in range and the clamp never fires; see the
    note on the kernel's domain at the top of this module for why it is here
    rather than an out-of-bounds read.
    """

    return table[value if value < len(table) else len(table) - 1]


@dataclass(frozen=True)
class _LutStage:
    tables: tuple[tuple[int, ...], ...]

    #: A lookup reads only its own channel, so one plane can run it alone.
    channel_groups = INDEPENDENT_GROUPS

    def evaluate(self, pixel: Pixel) -> Pixel:
        return tuple(  # type: ignore[return-value]
            _lookup(self.tables[channel], pixel[channel]) for channel in range(4)
        )


@dataclass(frozen=True)
class _LevelsStage:
    coefficients: tuple[float, ...]
    input_offsets: tuple[int, ...]
    output_offsets: tuple[int, ...]
    maximum: int
    #: Whether the host contracts the one multiply-add into a single rounding.
    #: True unless the operation says otherwise, which it only does when the
    #: two evaluations disagree somewhere in the format's domain.
    fused: bool = True

    #: The matrix is diagonal, so each channel depends only on itself.
    channel_groups = INDEPENDENT_GROUPS

    def evaluate(self, pixel: Pixel) -> Pixel:
        # Contracted, the whole expression rounds once. Both inputs have few
        # enough significant bits for binary64 to hold the exact intermediate
        # before that single explicit binary32 rounding.
        output = []
        for channel in range(4):
            product = (pixel[channel] - self.input_offsets[channel]) * self.coefficients[
                channel
            ]
            if not self.fused:
                product = _f32(product)
            value = _f32(product + self.output_offsets[channel])
            output.append(_saturate(int(value), self.maximum))
        return tuple(output)  # type: ignore[return-value]


@dataclass(frozen=True)
class _MixerStage:
    tables: tuple[tuple[tuple[int, ...], ...], ...]
    maximum: int

    #: Every output channel sums terms from all four inputs.
    channel_groups = WHOLE_PIXEL_GROUPS

    def evaluate(self, pixel: Pixel) -> Pixel:
        output = []
        for output_channel in range(4):
            terms = self.tables[output_channel]
            value = sum(
                _lookup(terms[input_channel], pixel[input_channel])
                for input_channel in range(4)
            )
            output.append(_saturate(value, self.maximum))
        return tuple(output)  # type: ignore[return-value]


@dataclass(frozen=True)
class _ChromaRotateStage:
    """``vf_hue.c``'s chroma rotation, as exact integer arithmetic.

    Upstream materializes this as two 64 KiB tables indexed by the ``(u, v)``
    pair.  The arithmetic behind them is a handful of 32-bit operations that
    cannot overflow at these magnitudes, so it is evaluated directly instead:
    same bytes, no table.
    """

    cosine: int
    sine: int
    #: Depth of the chroma samples: 8 for ``apply_lut``, 10 for ``apply_lut10``.
    depth: int = 8

    #: Chroma is a 2D vector here, so the two channels rotate into each other.
    channel_groups = CHROMA_PAIR_GROUPS

    def evaluate(self, pixel: Pixel) -> Pixel:
        maximum = (1 << self.depth) - 1
        centre = 1 << (self.depth - 1)
        # apply_lut10 clamps each sample into the depth before indexing its
        # table, which apply_lut has no need to do because a byte is always in
        # range. Reproducing the clamp is what makes the two paths one stage.
        u = _saturate(pixel[1], maximum) - centre
        v = _saturate(pixel[2], maximum) - centre
        # + (1 << 15) rounds the >> 16 to nearest; + (centre << 16) re-centres.
        rounding = (1 << 15) + (centre << 16)
        rotated_u = ((self.cosine * u) - (self.sine * v) + rounding) >> 16
        rotated_v = ((self.sine * u) + (self.cosine * v) + rounding) >> 16
        return (
            pixel[0],
            _saturate(rotated_u, maximum),
            _saturate(rotated_v, maximum),
            pixel[3],
        )


@dataclass(frozen=True)
class _ExprStage:
    """One straight-line float32 expression over the whole pixel."""

    program: ExprProgram

    @property
    def channel_groups(self) -> tuple[tuple[int, ...], ...]:
        """Every channel the program touches, as one group.

        An expression reads across channels freely, so all of them have to be
        in hand at once; a layout that samples them at different resolutions
        cannot supply that and :func:`_check_sampling_groups` refuses it.
        """

        touched = self.program.channels_read | self.program.channels_written
        return (tuple(sorted(touched)),)

    def evaluate(self, pixel: Pixel) -> Pixel:
        return self.program.evaluate(pixel)


Stage = _LutStage | _LevelsStage | _MixerStage | _ChromaRotateStage | _ExprStage


def _prepare_lut(operation: Operation, depth: int) -> _LutStage:
    name = lut_operation(depth)
    expected = {"tables"} if depth == 8 else {"tables", "depth"}
    if set(operation.parameters) != expected:
        raise InterpreterError(
            f"{name} accepts only the tables parameter"
            if depth == 8
            else f"{name} accepts only the tables and depth parameters"
        )
    if depth != 8 and operation.parameters["depth"] != depth:
        raise InterpreterError(
            f"{name} declares depth {operation.parameters['depth']!r}, but the "
            f"layout stores {depth}-bit samples"
        )
    size = 1 << depth
    maximum = size - 1
    tables_value = _sequence(operation.parameters.get("tables"), 4, f"{name} tables")
    tables: list[tuple[int, ...]] = []
    for channel, table_value in enumerate(tables_value):
        entries = _sequence(table_value, size, f"{name} channel {channel} table")
        table = tuple(_integer(entry, f"{name} table entry") for entry in entries)
        if any(not 0 <= entry <= maximum for entry in table):
            raise InterpreterError(f"{name} table entries must be in [0, {maximum}]")
        tables.append(table)
    return _LutStage(tuple(tables))


def _prepare_levels(operation: Operation, depth: int) -> _LevelsStage:
    parameters = operation.parameters
    expected_parameters = {
        "evaluation",
        "coefficients",
        "offsets",
        "input_max",
        "output_max",
    }
    if depth != 8:
        expected_parameters.add("depth")
    # "contraction" is present only when the contracted and separate
    # evaluations disagree somewhere, so its absence is itself a claim: this
    # operation means the same bytes on every host.
    contraction = parameters.get("contraction")
    if contraction is not None:
        if contraction not in {"fused", "separate"}:
            raise InterpreterError(
                f"unsupported levels contraction {contraction!r}"
            )
        expected_parameters.add("contraction")
    if set(parameters) != expected_parameters:
        raise InterpreterError(
            "levels_f32_fma parameters must be evaluation, coefficients, "
            "offsets, input_max, and output_max"
            + (", and depth" if depth != 8 else "")
        )
    if depth != 8 and parameters["depth"] != depth:
        raise InterpreterError(
            f"levels_f32_fma declares depth {parameters['depth']!r}, but the "
            f"layout stores {depth}-bit samples"
        )
    matrix = _sequence(parameters.get("coefficients"), 4, "levels coefficient matrix")
    coefficients: list[float] = []
    for row_index, row_value in enumerate(matrix):
        row = _sequence(row_value, 4, f"levels coefficient row {row_index}")
        for column, encoded in enumerate(row):
            if not isinstance(encoded, str):
                raise InterpreterError("levels coefficients must be hexadecimal strings")
            try:
                value = float.fromhex(encoded)
            except ValueError as error:
                raise InterpreterError(f"invalid levels coefficient {encoded!r}") from error
            if not math.isfinite(value):
                raise InterpreterError("levels coefficients must be finite")
            if row_index == column:
                coefficients.append(_f32(value))
            elif value != 0.0:
                raise InterpreterError("levels_f32_fma only accepts a diagonal matrix")

    offsets = _sequence(parameters.get("offsets"), 4, "levels offsets")
    input_offsets: list[int] = []
    output_offsets: list[int] = []
    for value in offsets:
        if not isinstance(value, dict) or set(value) != {"input", "output"}:
            raise InterpreterError("each levels offset must contain input and output")
        input_offsets.append(_integer(value["input"], "levels input offset"))
        output_offsets.append(_integer(value["output"], "levels output offset"))

    input_max = tuple(
        _integer(value, "levels input maximum")
        for value in _sequence(parameters["input_max"], 4, "levels input maxima")
    )
    output_max = tuple(
        _integer(value, "levels output maximum")
        for value in _sequence(parameters["output_max"], 4, "levels output maxima")
    )
    # Upstream scales the option's [0, 1] endpoint by UINT8_MAX at one byte per
    # sample and by UINT16_MAX at two, whatever the actual depth is, so a
    # 10-bit endpoint runs to 65535 while the stored result still clips to
    # 1023. That is a quirk of vf_colorlevels.c, reproduced rather than fixed.
    endpoint_max = 255 if depth == 8 else 65535
    for channel in range(4):
        if input_max[channel] == input_offsets[channel]:
            raise InterpreterError("levels input endpoints must not be equal")
        if not all(
            0 <= endpoint <= endpoint_max
            for endpoint in (
                input_offsets[channel],
                input_max[channel],
                output_offsets[channel],
                output_max[channel],
            )
        ):
            raise InterpreterError(f"levels endpoints must be in [0, {endpoint_max}]")
        expected = _f32(
            (output_max[channel] - output_offsets[channel])
            / (input_max[channel] - input_offsets[channel])
        )
        if coefficients[channel] != expected:
            raise InterpreterError("levels coefficient does not match its endpoints")
    return _LevelsStage(
        tuple(coefficients),
        tuple(input_offsets),
        tuple(output_offsets),
        (1 << depth) - 1,
        contraction != "separate",
    )


def _prepare_mixer(operation: Operation, depth: int) -> _MixerStage:
    parameters = operation.parameters
    evaluation = parameters.get("evaluation")
    folded = evaluation == "sum_i32_lut_terms"
    expected_parameters = (
        {"evaluation", "offsets", "contribution_tables"}
        if folded
        else {"evaluation", "coefficients", "offsets", "contribution_tables"}
    )
    if depth != 8:
        expected_parameters = expected_parameters | {"depth"}
        if parameters.get("depth") != depth:
            raise InterpreterError(
                f"mixer declares depth {parameters.get('depth')!r}, but the "
                f"layout stores {depth}-bit samples"
            )
    if set(parameters) != expected_parameters:
        if folded:
            raise InterpreterError(
                "folded mixer parameters must be evaluation, offsets, "
                "and contribution_tables"
            )
        raise InterpreterError(
            "mixer parameters must be evaluation, coefficients, offsets, "
            "and contribution_tables"
        )
    offsets = _sequence(parameters["offsets"], 4, "mixer offsets")
    if any(_integer(value, "mixer offset") != 0 for value in offsets):
        raise InterpreterError("mixer offsets must be zero")
    coefficients: list[tuple[float, ...]] | None = None
    if not folded:
        coefficient_rows = _sequence(
            parameters["coefficients"], 4, "mixer coefficients"
        )
        coefficients = []
        for row_value in coefficient_rows:
            row = _sequence(row_value, 4, "mixer coefficient row")
            values: list[float] = []
            for encoded in row:
                if not isinstance(encoded, str):
                    raise InterpreterError(
                        "mixer coefficients must be hexadecimal strings"
                    )
                try:
                    coefficient = float.fromhex(encoded)
                except ValueError as error:
                    raise InterpreterError(
                        f"invalid mixer coefficient {encoded!r}"
                    ) from error
                if not math.isfinite(coefficient) or not -2.0 <= coefficient <= 2.0:
                    raise InterpreterError(
                        "mixer coefficients must be finite and in [-2, 2]"
                    )
                values.append(coefficient)
            coefficients.append(tuple(values))

    size = 1 << depth
    # A term is at most twice the widest sample, either way round.
    term_bound = 2 * (size - 1)
    rows = _sequence(parameters["contribution_tables"], 4, "mixer table rows")
    prepared_rows: list[tuple[tuple[int, ...], ...]] = []
    for output_channel, row_value in enumerate(rows):
        row = _sequence(row_value, 4, f"mixer output row {output_channel}")
        tables: list[tuple[int, ...]] = []
        for input_channel, table_value in enumerate(row):
            entries = _sequence(
                table_value,
                size,
                f"mixer table {output_channel},{input_channel}",
            )
            table = tuple(_integer(entry, "mixer table entry") for entry in entries)
            if folded:
                if any(not -term_bound <= entry <= term_bound for entry in table):
                    raise InterpreterError(
                        "folded mixer contribution entries must be in "
                        f"[-{term_bound}, {term_bound}]"
                    )
            else:
                assert coefficients is not None
                expected = tuple(
                    round(value * coefficients[output_channel][input_channel])
                    for value in range(size)
                )
                if table != expected:
                    raise InterpreterError(
                        "mixer contribution table does not match coefficient"
                    )
            tables.append(table)
        prepared_rows.append(tuple(tables))
    return _MixerStage(tuple(prepared_rows), size - 1)


#: Widest ``hue`` coefficient: ``lrint(1.0 * (1 << 16) * 10)``, the saturation
#: bound.  Anything wider could overflow the int32 the arithmetic assumes.
_CHROMA_COEFFICIENT_LIMIT = 10 * (1 << 16)


def _prepare_chroma_rotate(operation: Operation, depth: int) -> _ChromaRotateStage:
    parameters = operation.parameters
    expected = {"cosine", "sine"} if depth == 8 else {"cosine", "sine", "depth"}
    if set(parameters) != expected:
        raise InterpreterError(
            "chroma_rotate_i32 parameters must be cosine and sine"
            + (" and depth" if depth != 8 else "")
        )
    if depth != 8 and parameters["depth"] != depth:
        raise InterpreterError(
            f"chroma_rotate_i32 declares depth {parameters['depth']!r}, but the "
            f"layout stores {depth}-bit samples"
        )
    cosine = _integer(parameters["cosine"], "chroma rotation cosine")
    sine = _integer(parameters["sine"], "chroma rotation sine")
    for value in (cosine, sine):
        if not -_CHROMA_COEFFICIENT_LIMIT <= value <= _CHROMA_COEFFICIENT_LIMIT:
            raise InterpreterError(
                "chroma rotation coefficients must be within "
                f"+/-{_CHROMA_COEFFICIENT_LIMIT}"
            )
    return _ChromaRotateStage(cosine, sine, depth)


def _prepare_expr(operation: Operation, depth: int) -> _ExprStage:
    if set(operation.parameters) != {"program"}:
        raise InterpreterError("expr_f32 accepts only the program parameter")
    try:
        program = ExprProgram.from_dict(operation.parameters["program"])
    except ExprError as error:
        raise InterpreterError(f"invalid expr_f32 program: {error}") from error
    for channel, output in enumerate(program.outputs):
        if output is not None and not output.quantize.endswith(f"_u{depth}"):
            raise InterpreterError(
                f"expr_f32 output {channel} quantizes to {output.quantize!r}, "
                f"which is not the layout's {depth}-bit width"
            )
    return _ExprStage(program)


def _prepare(ir: PixelIR) -> tuple[Stage, ...]:
    if ir.ir_version != IR_VERSION:
        raise InterpreterError(
            f"unsupported IR version {ir.ir_version}; expected {IR_VERSION}"
        )
    if ir.layout not in LAYOUTS:
        raise InterpreterError(f"unsupported pixel layout {ir.layout!r}")
    depth = LAYOUTS[ir.layout].depth
    if ir.pixel_format != pixel_format_for(depth):
        raise InterpreterError(
            f"pixel format {ir.pixel_format!r} does not describe the "
            f"{depth}-bit samples of layout {ir.layout!r}"
        )
    load, store = load_operation(depth), store_operation(depth)
    quantize_kind = quantize_operation(depth)
    lut_kind = lut_operation(depth)
    lookup_mode = "lookup_u8" if depth == 8 else "lookup_u16"
    saturate_mode = "saturate_i32_to_u8" if depth == 8 else "saturate_i32_to_u16"
    supported_modes = _SUPPORTED_QUANTIZERS[8 if depth == 8 else 16]

    if len(ir.operations) < 2:
        raise InterpreterError("IR must contain a load and a store")
    if ir.operations[0].kind != load or ir.operations[-1].kind != store:
        raise InterpreterError(f"IR must begin with {load} and end with {store}")
    if ir.operations[0].parameters or ir.operations[-1].parameters:
        raise InterpreterError(f"{load} and {store} do not accept parameters")

    body = ir.operations[1:-1]
    if len(body) % 2:
        raise InterpreterError(f"each transform must be followed by {quantize_kind}")
    stages: list[Stage] = []
    for index in range(0, len(body), 2):
        transform = body[index]
        quantize = body[index + 1]
        if quantize.kind != quantize_kind:
            raise InterpreterError(
                f"operation {index + 2} must be {quantize_kind}, got {quantize.kind!r}"
            )
        mode = quantize.parameters.get("mode")
        if set(quantize.parameters) != {"mode"} or mode not in supported_modes:
            raise InterpreterError(f"unsupported quantization mode {mode!r}")

        if transform.kind == lut_kind:
            if mode not in {lookup_mode, "truncate_toward_zero_then_saturate"}:
                raise InterpreterError(
                    f"{lut_kind} cannot use quantization mode {mode!r}"
                )
            stages.append(_prepare_lut(transform, depth))
        elif transform.kind == "expr_f32":
            if mode != "expression_outputs":
                raise InterpreterError(
                    f"expr_f32 cannot use quantization mode {mode!r}"
                )
            stages.append(_prepare_expr(transform, depth))
        elif transform.kind == "chroma_rotate_i32":
            if mode != saturate_mode:
                raise InterpreterError(
                    f"chroma_rotate_i32 cannot use quantization mode {mode!r}"
                )
            stages.append(_prepare_chroma_rotate(transform, depth))
        elif transform.kind == "matrix4x4":
            evaluation = transform.parameters.get("evaluation")
            if evaluation == "levels_f32_fma":
                if mode != "truncate_toward_zero_then_saturate":
                    raise InterpreterError(
                        f"levels_f32_fma cannot use quantization mode {mode!r}"
                    )
                stages.append(_prepare_levels(transform, depth))
            elif evaluation in {
                "sum_i32_terms_rounded_ties_even",
                "sum_i32_lut_terms",
            }:
                if mode != saturate_mode:
                    raise InterpreterError(
                        f"mixer contribution tables cannot use quantization mode {mode!r}"
                    )
                stages.append(_prepare_mixer(transform, depth))
            else:
                raise InterpreterError(f"unsupported matrix evaluation {evaluation!r}")
        else:
            raise InterpreterError(f"unsupported transform {transform.kind!r}")

    _check_sampling_groups(ir.layout, stages)
    return tuple(stages)


def _check_sampling_groups(name: str, stages: list[Stage]) -> None:
    """Refuse a stage that reads across channels with no common sample grid.

    A subsampled layout stores one chroma sample per several luma samples, so
    an operation mixing luma with chroma has no single pixel to mix.  The
    admissible mixes are exactly those confined to one of the layout's sampling
    groups: ``yuv420p`` can rotate Cb into Cr, because those two are sampled at
    the same positions, but cannot fold luma into either.

    Every consumer of the IR reaches this check, so the restriction cannot be
    bypassed by constructing the IR directly.
    """

    layout = LAYOUTS[name]
    stored = set(layout.stored_channels)
    groups = [set(group) for group in layout.sampling_groups]
    for index, stage in enumerate(stages):
        for channels in stage.channel_groups:
            required = set(channels) & stored
            if len(required) > 1 and not any(
                required <= group for group in groups
            ):
                raise InterpreterError(
                    f"stage {index} reads across channels "
                    f"{sorted(required)} together, which the layout {name!r} "
                    "samples at different resolutions"
                )


def _validate_pixel(pixel: tuple[int, ...] | list[int], maximum: int) -> Pixel:
    values = _sequence(pixel, 4, "pixel")
    prepared = tuple(_integer(value, "pixel channel") for value in values)
    if any(not 0 <= value <= maximum for value in prepared):
        raise InterpreterError(f"pixel channels must be in [0, {maximum}]")
    return prepared  # type: ignore[return-value]


def _run_pixel(stages: tuple[Stage, ...], pixel: Pixel) -> Pixel:
    for stage in stages:
        pixel = stage.evaluate(pixel)
    return pixel


def interpret_pixel(ir: PixelIR, pixel: tuple[int, ...] | list[int]) -> Pixel:
    """Interpret one pixel and return its four output channels."""

    return _run_pixel(
        _prepare(ir), _validate_pixel(pixel, get_layout(ir.layout).max_value)
    )


def validate_ir(ir: PixelIR) -> None:
    """Validate that *ir* has the executable reference shape."""

    _prepare(ir)


def _byte_view(value: Any, description: str, *, writable: bool) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise InterpreterError(f"{description} must support the buffer protocol") from error
    if writable and view.readonly:
        raise InterpreterError(f"{description} must be writable")
    if not view.c_contiguous:
        raise InterpreterError(f"{description} must be C-contiguous")
    try:
        return view.cast("B")
    except TypeError as error:
        raise InterpreterError(f"{description} must have a byte-compatible layout") from error


def _validate_layout(
    length: int,
    offset: int,
    stride: int,
    row_bytes: int,
    height: int,
    description: str,
) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise InterpreterError(f"{description} offset must be an integer")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise InterpreterError(f"{description} stride must be an integer")
    first = offset
    last = offset + (height - 1) * stride
    lowest = min(first, last)
    highest = max(first, last) + row_bytes
    if lowest < 0 or highest > length:
        raise InterpreterError(
            f"{description} buffer is too small for offset={offset}, stride={stride}, "
            f"row_bytes={row_bytes}, height={height}"
        )


def _plane_strides(
    layout: PixelLayout,
    width: int,
    value: int | Sequence[int] | None,
    description: str,
) -> tuple[int, ...]:
    """Resolve a stride argument into one stride per plane.

    ``None`` means a tightly packed frame, where each plane's stride is its own
    row length; a single integer applies to every plane, which is what a packed
    layout has always meant.
    """

    if value is None:
        return tuple(
            layout.plane_row_bytes(plane, width)
            for plane in range(layout.plane_count)
        )
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * layout.plane_count
    if isinstance(value, Sequence) and len(value) == layout.plane_count:
        return tuple(
            _integer(entry, f"{description} stride") for entry in value
        )
    raise InterpreterError(
        f"{description} stride must be an integer or one stride per plane "
        f"({layout.plane_count})"
    )


def _sample_reader(view: memoryview, sample_bytes: int) -> Any:
    """Return a function reading one sample at a byte offset.

    Above eight bits a sample is a little-endian pair, which is what every
    layout in the table stores; see the module docstring on why the big-endian
    members are not listed.
    """

    if sample_bytes == 1:
        return lambda index: view[index]
    return lambda index: view[index] | (view[index + 1] << 8)


def _sample_writer(view: memoryview, sample_bytes: int) -> Any:
    if sample_bytes == 1:

        def write_byte(index: int, value: int) -> None:
            view[index] = value

        return write_byte

    def write_word(index: int, value: int) -> None:
        view[index] = value & 0xFF
        view[index + 1] = value >> 8

    return write_word


def interpret_into(
    ir: PixelIR,
    source: Any,
    destination: Any,
    width: int,
    height: int,
    *,
    source_stride: int | Sequence[int] | None = None,
    destination_stride: int | Sequence[int] | None = None,
    source_offset: int = 0,
    destination_offset: int = 0,
) -> None:
    """Interpret a frame into a distinct destination buffer.

    ``width`` and ``height`` describe plane 0; a subsampled plane derives its
    own dimensions from the layout. Strides are measured in bytes and may be
    negative when the corresponding offset points at the first logical row of
    every plane. Row padding is never read or modified.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise InterpreterError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise InterpreterError("height must be a positive integer")
    layout = get_layout(ir.layout)
    source_strides = _plane_strides(layout, width, source_stride, "source")
    destination_strides = _plane_strides(
        layout, width, destination_stride, "destination"
    )
    source_view = _byte_view(source, "source", writable=False)
    destination_view = _byte_view(destination, "destination", writable=True)
    if source_view.obj is destination_view.obj:
        raise InterpreterError("source and destination must use distinct buffers")
    # Planes of a tightly packed frame sit back to back, which is the layout
    # FFmpeg's rawvideo muxer writes and the one callers pass in.
    for plane in range(layout.plane_count):
        origin = layout.plane_origin(plane, width, height)
        for label, view, offset, strides in (
            ("source", source_view, source_offset, source_strides),
            ("destination", destination_view, destination_offset, destination_strides),
        ):
            _validate_layout(
                len(view),
                offset + origin,
                strides[plane],
                layout.plane_row_bytes(plane, width),
                layout.plane_height(plane, height),
                f"{label} plane {plane}",
            )

    stages = _prepare(ir)
    if layout.subsampled:
        _interpret_groups(
            stages,
            layout,
            source_view,
            destination_view,
            width,
            height,
            source_strides,
            destination_strides,
            source_offset,
            destination_offset,
        )
        return

    # Every plane shares the frame's resolution here, so one pixel is one
    # iteration and a stage may read all four channels of it.
    step = layout.step * layout.sample_bytes
    read_sample = _sample_reader(source_view, layout.sample_bytes)
    write_sample = _sample_writer(destination_view, layout.sample_bytes)
    stored = layout.stored_channels
    origins = [
        layout.plane_origin(plane, width, height)
        for plane in range(layout.plane_count)
    ]
    # Per channel: the plane it lives in and the byte offset of its sample
    # inside a group.
    reads = tuple(
        None
        if layout.plane(channel) is None
        else (layout.plane(channel), layout.offset(channel) * layout.sample_bytes)
        for channel in range(4)
    )
    for y in range(height):
        source_rows = [
            source_offset + origins[plane] + y * source_strides[plane]
            for plane in range(layout.plane_count)
        ]
        destination_rows = [
            destination_offset + origins[plane] + y * destination_strides[plane]
            for plane in range(layout.plane_count)
        ]
        for x in range(width):
            # An absent alpha loads as zero, matching upstream's have_alpha=0 path.
            pixel: Pixel = tuple(  # type: ignore[assignment]
                0
                if reads[channel] is None
                else read_sample(
                    source_rows[reads[channel][0]] + x * step + reads[channel][1]
                )
                for channel in range(4)
            )
            output = _run_pixel(stages, pixel)
            for channel in stored:
                plane, offset = reads[channel]  # type: ignore[misc]
                write_sample(
                    destination_rows[plane] + x * step + offset, output[channel]
                )


def _interpret_groups(
    stages: tuple[Stage, ...],
    layout: PixelLayout,
    source_view: memoryview,
    destination_view: memoryview,
    width: int,
    height: int,
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
    source_offset: int,
    destination_offset: int,
) -> None:
    """Run the pipeline one sampling group at a time.

    A subsampled layout has no single iteration space, so each group of
    co-sited channels is walked at its own resolution.  Channels outside the
    group are loaded as zero and their results discarded: :func:`_prepare` has
    already refused any stage whose output inside this group would have needed
    them.
    """

    read_sample = _sample_reader(source_view, layout.sample_bytes)
    write_sample = _sample_writer(destination_view, layout.sample_bytes)
    for channels in layout.sampling_groups:
        planes = tuple(layout.plane(channel) for channel in channels)
        offsets = tuple(
            layout.offset(channel) * layout.sample_bytes  # type: ignore[operator]
            for channel in channels
        )
        # Every channel of a group shares one resolution, by construction.
        group_width = layout.plane_width(planes[0], width)  # type: ignore[arg-type]
        group_height = layout.plane_height(planes[0], height)  # type: ignore[arg-type]
        origins = tuple(
            layout.plane_origin(plane, width, height)  # type: ignore[arg-type]
            for plane in planes
        )
        for y in range(group_height):
            source_rows = tuple(
                source_offset + origin + y * source_strides[plane]  # type: ignore[index]
                for plane, origin in zip(planes, origins, strict=True)
            )
            destination_rows = tuple(
                destination_offset + origin + y * destination_strides[plane]  # type: ignore[index]
                for plane, origin in zip(planes, origins, strict=True)
            )
            for x in range(group_width):
                indices = tuple(
                    x * layout.step * layout.sample_bytes + offset
                    for offset in offsets
                )
                loaded = [0, 0, 0, 0]
                for position, channel in enumerate(channels):
                    loaded[channel] = read_sample(
                        source_rows[position] + indices[position]
                    )
                pixel: Pixel = tuple(loaded)  # type: ignore[assignment]
                output = _run_pixel(stages, pixel)
                for position, channel in enumerate(channels):
                    write_sample(
                        destination_rows[position] + indices[position],
                        output[channel],
                    )


def interpret_rgba8(ir: PixelIR, source: Any, width: int, height: int) -> bytes:
    """Interpret one tightly packed frame in the IR's own layout."""

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise InterpreterError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise InterpreterError("height must be a positive integer")
    expected = get_layout(ir.layout).frame_size(width, height)
    source_view = _byte_view(source, "source", writable=False)
    if len(source_view) != expected:
        raise InterpreterError(
            f"packed source has {len(source_view)} bytes; expected {expected}"
        )
    output = bytearray(expected)
    interpret_into(ir, source_view, output, width, height)
    return bytes(output)
