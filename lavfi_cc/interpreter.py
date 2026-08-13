"""Deliberately simple scalar interpreter for the RGBA8 pixel IR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import struct
from typing import Any

from .ir import IR_VERSION, Operation, PixelIR
from .layouts import LAYOUTS, PixelLayout, get_layout


class InterpreterError(ValueError):
    """The IR or frame layout cannot be interpreted safely."""


Pixel = tuple[int, int, int, int]
_SUPPORTED_QUANTIZERS = {
    "lookup_u8",
    "truncate_toward_zero_then_saturate",
    "saturate_i32_to_u8",
}


def _f32(value: float) -> float:
    """Round one operation result to IEEE-754 binary32."""

    return struct.unpack("=f", struct.pack("=f", value))[0]


def _u8(value: int) -> int:
    return max(0, min(255, value))


def _sequence(value: Any, length: int, description: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InterpreterError(f"{description} must contain {length} entries")
    return tuple(value)


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InterpreterError(f"{description} must be an integer")
    return value


@dataclass(frozen=True)
class _LutStage:
    tables: tuple[tuple[int, ...], ...]

    #: A lookup reads only its own channel, so one plane can run it alone.
    channel_independent = True

    def evaluate_channel(self, channel: int, value: int) -> int:
        return self.tables[channel][value]

    def evaluate(self, pixel: Pixel) -> Pixel:
        return tuple(  # type: ignore[return-value]
            self.tables[channel][pixel[channel]] for channel in range(4)
        )


@dataclass(frozen=True)
class _LevelsStage:
    coefficients: tuple[float, ...]
    input_offsets: tuple[int, ...]
    output_offsets: tuple[int, ...]

    #: The matrix is diagonal, so each channel depends only on itself.
    channel_independent = True

    def evaluate_channel(self, channel: int, value: int) -> int:
        # The pinned Apple-Clang FFmpeg oracle contracts this expression to one
        # binary32 FMLA. Both inputs have few enough significant bits for
        # binary64 to hold the exact intermediate before the single explicit
        # binary32 rounding below.
        return _u8(
            int(
                _f32(
                    (value - self.input_offsets[channel])
                    * self.coefficients[channel]
                    + self.output_offsets[channel]
                )
            )
        )

    def evaluate(self, pixel: Pixel) -> Pixel:
        return tuple(  # type: ignore[return-value]
            self.evaluate_channel(channel, pixel[channel]) for channel in range(4)
        )


@dataclass(frozen=True)
class _MixerStage:
    tables: tuple[tuple[tuple[int, ...], ...], ...]

    #: Every output channel sums terms from all four inputs.
    channel_independent = False

    def evaluate(self, pixel: Pixel) -> Pixel:
        output = []
        for output_channel in range(4):
            terms = self.tables[output_channel]
            value = sum(
                terms[input_channel][pixel[input_channel]]
                for input_channel in range(4)
            )
            output.append(_u8(value))
        return tuple(output)  # type: ignore[return-value]


Stage = _LutStage | _LevelsStage | _MixerStage


def _prepare_lut(operation: Operation) -> _LutStage:
    if set(operation.parameters) != {"tables"}:
        raise InterpreterError("lut8 accepts only the tables parameter")
    tables_value = _sequence(operation.parameters.get("tables"), 4, "lut8 tables")
    tables: list[tuple[int, ...]] = []
    for channel, table_value in enumerate(tables_value):
        entries = _sequence(table_value, 256, f"lut8 channel {channel} table")
        table = tuple(_integer(entry, "lut8 table entry") for entry in entries)
        if any(not 0 <= entry <= 255 for entry in table):
            raise InterpreterError("lut8 table entries must be in [0, 255]")
        tables.append(table)
    return _LutStage(tuple(tables))


def _prepare_levels(operation: Operation) -> _LevelsStage:
    parameters = operation.parameters
    expected_parameters = {
        "evaluation",
        "coefficients",
        "offsets",
        "input_max",
        "output_max",
    }
    if set(parameters) != expected_parameters:
        raise InterpreterError(
            "levels_f32_fma parameters must be evaluation, coefficients, "
            "offsets, input_max, and output_max"
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
    for channel in range(4):
        if input_max[channel] == input_offsets[channel]:
            raise InterpreterError("levels input endpoints must not be equal")
        if not all(
            0 <= endpoint <= 255
            for endpoint in (
                input_offsets[channel],
                input_max[channel],
                output_offsets[channel],
                output_max[channel],
            )
        ):
            raise InterpreterError("levels endpoints must be in [0, 255]")
        expected = _f32(
            (output_max[channel] - output_offsets[channel])
            / (input_max[channel] - input_offsets[channel])
        )
        if coefficients[channel] != expected:
            raise InterpreterError("levels coefficient does not match its endpoints")
    return _LevelsStage(tuple(coefficients), tuple(input_offsets), tuple(output_offsets))


def _prepare_mixer(operation: Operation) -> _MixerStage:
    parameters = operation.parameters
    evaluation = parameters.get("evaluation")
    folded = evaluation == "sum_i32_lut_terms"
    expected_parameters = (
        {"evaluation", "offsets", "contribution_tables"}
        if folded
        else {"evaluation", "coefficients", "offsets", "contribution_tables"}
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

    rows = _sequence(parameters["contribution_tables"], 4, "mixer table rows")
    prepared_rows: list[tuple[tuple[int, ...], ...]] = []
    for output_channel, row_value in enumerate(rows):
        row = _sequence(row_value, 4, f"mixer output row {output_channel}")
        tables: list[tuple[int, ...]] = []
        for input_channel, table_value in enumerate(row):
            entries = _sequence(
                table_value,
                256,
                f"mixer table {output_channel},{input_channel}",
            )
            table = tuple(_integer(entry, "mixer table entry") for entry in entries)
            if folded:
                if any(not -510 <= entry <= 510 for entry in table):
                    raise InterpreterError(
                        "folded mixer contribution entries must be in [-510, 510]"
                    )
            else:
                assert coefficients is not None
                expected = tuple(
                    round(value * coefficients[output_channel][input_channel])
                    for value in range(256)
                )
                if table != expected:
                    raise InterpreterError(
                        "mixer contribution table does not match coefficient"
                    )
            tables.append(table)
        prepared_rows.append(tuple(tables))
    return _MixerStage(tuple(prepared_rows))


def _prepare(ir: PixelIR) -> tuple[Stage, ...]:
    if ir.ir_version != IR_VERSION:
        raise InterpreterError(
            f"unsupported IR version {ir.ir_version}; expected {IR_VERSION}"
        )
    if ir.pixel_format != "rgba8":
        raise InterpreterError(f"unsupported pixel format {ir.pixel_format!r}")
    if ir.layout not in LAYOUTS:
        raise InterpreterError(f"unsupported pixel layout {ir.layout!r}")
    if len(ir.operations) < 2:
        raise InterpreterError("IR must contain a load and a store")
    if ir.operations[0].kind != "load_rgba8" or ir.operations[-1].kind != "store_rgba8":
        raise InterpreterError("IR must begin with load_rgba8 and end with store_rgba8")
    if ir.operations[0].parameters or ir.operations[-1].parameters:
        raise InterpreterError("load_rgba8 and store_rgba8 do not accept parameters")

    body = ir.operations[1:-1]
    if len(body) % 2:
        raise InterpreterError("each transform must be followed by quantize_rgba8")
    stages: list[Stage] = []
    for index in range(0, len(body), 2):
        transform = body[index]
        quantize = body[index + 1]
        if quantize.kind != "quantize_rgba8":
            raise InterpreterError(
                f"operation {index + 2} must be quantize_rgba8, got {quantize.kind!r}"
            )
        mode = quantize.parameters.get("mode")
        if set(quantize.parameters) != {"mode"} or mode not in _SUPPORTED_QUANTIZERS:
            raise InterpreterError(f"unsupported quantization mode {mode!r}")

        if transform.kind == "lut8":
            if mode not in {"lookup_u8", "truncate_toward_zero_then_saturate"}:
                raise InterpreterError(f"lut8 cannot use quantization mode {mode!r}")
            stages.append(_prepare_lut(transform))
        elif transform.kind == "matrix4x4":
            evaluation = transform.parameters.get("evaluation")
            if evaluation == "levels_f32_fma":
                if mode != "truncate_toward_zero_then_saturate":
                    raise InterpreterError(
                        f"levels_f32_fma cannot use quantization mode {mode!r}"
                    )
                stages.append(_prepare_levels(transform))
            elif evaluation in {
                "sum_i32_terms_rounded_ties_even",
                "sum_i32_lut_terms",
            }:
                if mode != "saturate_i32_to_u8":
                    raise InterpreterError(
                        f"mixer contribution tables cannot use quantization mode {mode!r}"
                    )
                stages.append(_prepare_mixer(transform))
            else:
                raise InterpreterError(f"unsupported matrix evaluation {evaluation!r}")
        else:
            raise InterpreterError(f"unsupported transform {transform.kind!r}")

    # A subsampled layout stores one chroma sample per several luma samples, so
    # there is no single pixel whose four channels an operation could mix. Every
    # consumer of the IR reaches this check, so the restriction cannot be
    # bypassed by constructing the IR directly.
    if LAYOUTS[ir.layout].subsampled:
        for index, stage in enumerate(stages):
            if not stage.channel_independent:
                raise InterpreterError(
                    f"stage {index} mixes channels, which the subsampled layout "
                    f"{ir.layout!r} cannot express"
                )
    return tuple(stages)


def _validate_pixel(pixel: tuple[int, ...] | list[int]) -> Pixel:
    values = _sequence(pixel, 4, "pixel")
    prepared = tuple(_integer(value, "pixel channel") for value in values)
    if any(not 0 <= value <= 255 for value in prepared):
        raise InterpreterError("pixel channels must be in [0, 255]")
    return prepared  # type: ignore[return-value]


def _run_pixel(stages: tuple[Stage, ...], pixel: Pixel) -> Pixel:
    for stage in stages:
        pixel = stage.evaluate(pixel)
    return pixel


def interpret_pixel(ir: PixelIR, pixel: tuple[int, ...] | list[int]) -> Pixel:
    """Interpret one RGBA8 pixel and return its four output channels."""

    return _run_pixel(_prepare(ir), _validate_pixel(pixel))


def validate_ir(ir: PixelIR) -> None:
    """Validate that *ir* has the executable RGBA8 reference shape."""

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
        _interpret_planes(
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
    step = layout.step
    stored = layout.stored_channels
    origins = [
        layout.plane_origin(plane, width, height)
        for plane in range(layout.plane_count)
    ]
    # Per channel: the plane it lives in and its offset in a sample group.
    reads = tuple(
        None
        if layout.plane(channel) is None
        else (layout.plane(channel), layout.offset(channel))
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
                else source_view[
                    source_rows[reads[channel][0]] + x * step + reads[channel][1]
                ]
                for channel in range(4)
            )
            output = _run_pixel(stages, pixel)
            for channel in stored:
                plane, offset = reads[channel]  # type: ignore[misc]
                destination_view[
                    destination_rows[plane] + x * step + offset
                ] = output[channel]


def _interpret_planes(
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
    """Run a channel-independent pipeline one plane at a time.

    A subsampled layout has no common iteration space, so each plane is walked
    at its own resolution and only the channels it stores are evaluated.
    :func:`_prepare` has already refused any stage that would need the others.
    """

    for plane, channels in enumerate(layout.plane_channels):
        origin = layout.plane_origin(plane, width, height)
        plane_width = layout.plane_width(plane, width)
        plane_height = layout.plane_height(plane, height)
        offsets = tuple(layout.offset(channel) for channel in channels)
        for y in range(plane_height):
            source_row = source_offset + origin + y * source_strides[plane]
            destination_row = (
                destination_offset + origin + y * destination_strides[plane]
            )
            for x in range(plane_width):
                for channel, offset in zip(channels, offsets, strict=True):
                    index = x * layout.step + offset  # type: ignore[operator]
                    value = source_view[source_row + index]
                    for stage in stages:
                        value = stage.evaluate_channel(channel, value)
                    destination_view[destination_row + index] = value


def interpret_rgba8(ir: PixelIR, source: Any, width: int, height: int) -> bytes:
    """Interpret one tightly packed RGBA8 frame."""

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
