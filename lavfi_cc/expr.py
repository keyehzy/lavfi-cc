"""Straight-line float32 expression programs over the four logical channels.

Every operation the pixel IR had before this one is a table or a matrix: each
output channel is a function of one input channel, or a sum of four independent
per-channel terms.  ``colorbalance`` and ``colorcontrast`` are neither.  Their
per-pixel lightness terms read all three colour channels and feed all three
back, through an expression whose exact float32 operation order is the
filter's observable behaviour::

    const float l = (FFMAX3(r, g, b) + FFMIN3(r, g, b));
    r = get_component(r, l, shadows, midtones, highlights);

An :class:`ExprProgram` is how the IR says that.  It is a single-assignment
list of float32 instructions -- no branches, no loops, no memory -- plus one
optional output per channel.  Besides arithmetic it carries exact comparisons
and intermediate ``lrintf`` for ``selectivecolor``, whose independently rounded
range terms cannot be recovered by one output quantizer.  A channel with no
output keeps the sample it came in with, which is what upstream does with
alpha.

**Why the operation order is spelled out.** Every instruction here rounds its
result to binary32, exactly once, and the order they appear in is the order
upstream evaluates them.  Reassociating ``(a + b) + c`` into ``a + (b + c)``
can change stored samples, so the program is a transcription rather than a
formula.  For the same reason ``min`` and ``max`` reproduce FFmpeg's
``FFMIN``/``FFMAX``
macros -- ternaries on ``>``, not ``fminf``/``fmaxf``, which disagree with them
on signed zeros.

**Why there is an explicit fma.** A C compiler is allowed to contract ``a * b +
c`` into a fused multiply-add, and at ``-ffp-contract=on`` -- the default, and
what the pinned oracle is built with -- Clang does so wherever the target has
the instruction.  That is not a rounding this compiler may ignore: over the
complete 256^3 RGB domain, ``colorcontrast`` produces 147 different bytes with
and without contraction.  So the IR states which multiply-adds are fused
instead of leaving it to whichever compiler sees the generated C, and the
kernel is compiled with ``-ffp-contract=off`` so nothing else can be fused
behind its back.  A lowerer that transcribes an upstream expression is
responsible for marking the sites upstream's compiler fuses; see
:func:`multiply_is_exact` for the case where the distinction provably cannot
matter.

The interpreter and the C generator both live in this module, next to each
other, because the only thing that makes them correct is that they agree
instruction for instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any


class ExprError(ValueError):
    """An expression program is malformed or cannot be evaluated."""


_PACK = struct.Struct("=f")


def f32(value: float) -> float:
    """Round one operation result to IEEE-754 binary32."""

    try:
        return _PACK.unpack(_PACK.pack(value))[0]  # type: ignore[no-any-return]
    except OverflowError:
        return math.inf if value > 0 else -math.inf


#: ``opcode -> number of operands``.  Every operand is an index into the
#: instructions already emitted, except ``channel`` (a logical channel number)
#: and ``const`` (a hexadecimal float32 literal).
OPCODES: dict[str, int] = {
    "channel": 1,
    "const": 1,
    "neg": 1,
    "abs": 1,
    "floor": 1,
    # Round to the nearest integer under the process rounding mode, preserving
    # the result as an exactly-representable float for later integer-style
    # accumulation.  selectivecolor rounds each configured range before it
    # adds the ranges together, so an output-only quantizer cannot express it.
    "lrintf": 1,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    # FFMIN and FFMAX, which are ternaries on ">" rather than libm's fminf and
    # fmaxf: the two disagree when both operands are zeros of opposite sign.
    "min": 2,
    "max": 2,
    # Comparisons produce the float32 values 0 or 1.  Keeping predicates in
    # the same straight-line value space lets a lowering mask a term without
    # adding control flow to the pixel program.
    "eq": 2,
    "gt": 2,
    # One fused multiply-add: fma(a, b, c) rounds a * b + c once.
    "fma": 3,
}

#: Component depths a stored sample can have, from the layout table.
SAMPLE_DEPTHS = (8, 9, 10, 12, 14, 16)

#: How a channel's final float becomes its stored sample.
#:
#: Each reproduces an ``av_clip_uint8`` -- or, above eight bits, the
#: ``av_clip_uintp2_c`` upstream switches to -- of a float that C converts to
#: ``int`` first.  Clamping in float ahead of the conversion is the same mapping
#: for every input C defines and leaves nothing undefined for the ones it does
#: not.  The name carries the depth because a program is otherwise
#: self-contained: nothing else in it says how wide its outputs are.
QUANTIZERS = tuple(
    f"{rule}_u{depth}"
    for depth in SAMPLE_DEPTHS
    for rule in ("lrintf_saturate", "truncate_saturate")
)


def quantizer_maximum(rule: str) -> int:
    """Largest value *rule* can produce."""

    return (1 << int(rule.rpartition("_u")[2])) - 1


def _hex_float32(value: float) -> str:
    rounded = f32(value)
    if not math.isfinite(rounded):
        raise ExprError(f"expression constants must be finite, got {value!r}")
    return rounded.hex()


def _parse_hex_float32(encoded: Any) -> float:
    if not isinstance(encoded, str):
        raise ExprError("expression constants must be hexadecimal strings")
    try:
        value = float.fromhex(encoded)
    except ValueError as error:
        raise ExprError(f"invalid expression constant {encoded!r}") from error
    if not math.isfinite(value):
        raise ExprError("expression constants must be finite")
    if f32(value) != value:
        raise ExprError(f"expression constant {encoded!r} is not a binary32 value")
    return value


@dataclass(frozen=True)
class Output:
    """One channel's result: which value to store, and how to quantize it."""

    value: int
    quantize: str

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "quantize": self.quantize}


@dataclass(frozen=True)
class ExprProgram:
    """A validated single-assignment float32 program and its channel outputs."""

    instructions: tuple[tuple[Any, ...], ...]
    outputs: tuple[Output | None, Output | None, Output | None, Output | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "instructions": [list(instruction) for instruction in self.instructions],
            "outputs": [
                None if output is None else output.as_dict() for output in self.outputs
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ExprProgram":
        if not isinstance(value, dict) or set(value) != {"instructions", "outputs"}:
            raise ExprError("an expression program needs instructions and outputs")
        raw_instructions = value["instructions"]
        if not isinstance(raw_instructions, (list, tuple)):
            raise ExprError("expression instructions must be a list")
        instructions: list[tuple[Any, ...]] = []
        for index, raw in enumerate(raw_instructions):
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ExprError(f"instruction {index} is not an opcode and operands")
            opcode = raw[0]
            arity = OPCODES.get(opcode) if isinstance(opcode, str) else None
            if arity is None:
                raise ExprError(f"unsupported expression opcode {opcode!r}")
            operands = tuple(raw[1:])
            if len(operands) != arity:
                raise ExprError(
                    f"opcode {opcode!r} takes {arity} operand"
                    f"{'s' if arity != 1 else ''}, got {len(operands)}"
                )
            if opcode == "channel":
                channel = operands[0]
                if isinstance(channel, bool) or not isinstance(channel, int):
                    raise ExprError("a channel operand must be an integer")
                if not 0 <= channel <= 3:
                    raise ExprError(f"channel {channel} is outside 0..3")
            elif opcode == "const":
                _parse_hex_float32(operands[0])
            else:
                for operand in operands:
                    if isinstance(operand, bool) or not isinstance(operand, int):
                        raise ExprError(f"instruction {index} operands must be integers")
                    # Single assignment, defined before use: no forward
                    # references means no cycles and one evaluation order.
                    if not 0 <= operand < index:
                        raise ExprError(
                            f"instruction {index} refers to {operand}, which is not "
                            "an earlier instruction"
                        )
            instructions.append((opcode, *operands))

        raw_outputs = value["outputs"]
        if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 4:
            raise ExprError("an expression program needs exactly four outputs")
        outputs: list[Output | None] = []
        for channel, raw_output in enumerate(raw_outputs):
            if raw_output is None:
                outputs.append(None)
                continue
            if not isinstance(raw_output, dict) or set(raw_output) != {
                "value",
                "quantize",
            }:
                raise ExprError(f"output {channel} needs a value and a quantize rule")
            target = raw_output["value"]
            if isinstance(target, bool) or not isinstance(target, int):
                raise ExprError(f"output {channel} value must be an integer")
            if not 0 <= target < len(instructions):
                raise ExprError(
                    f"output {channel} refers to instruction {target}, which does "
                    "not exist"
                )
            quantize = raw_output["quantize"]
            if quantize not in QUANTIZERS:
                raise ExprError(f"unsupported output quantizer {quantize!r}")
            outputs.append(Output(target, quantize))
        return cls(tuple(instructions), tuple(outputs))  # type: ignore[arg-type]

    @property
    def channels_read(self) -> frozenset[int]:
        return frozenset(
            instruction[1]
            for instruction in self.instructions
            if instruction[0] == "channel"
        )

    @property
    def channels_written(self) -> frozenset[int]:
        return frozenset(
            channel for channel, output in enumerate(self.outputs) if output is not None
        )

    @property
    def is_identity(self) -> bool:
        """Whether the program stores nothing, so every channel passes through."""

        return not self.channels_written

    def evaluate(self, pixel: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Run the program over four logical channels and return output samples."""

        values: list[float] = []
        for instruction in self.instructions:
            opcode = instruction[0]
            if opcode == "channel":
                values.append(float(pixel[instruction[1]]))
                continue
            if opcode == "const":
                values.append(float.fromhex(instruction[1]))
                continue
            if opcode == "neg":
                values.append(f32(-values[instruction[1]]))
                continue
            if opcode == "abs":
                values.append(f32(abs(values[instruction[1]])))
                continue
            if opcode == "floor":
                values.append(f32(math.floor(values[instruction[1]])))
                continue
            if opcode == "lrintf":
                values.append(f32(round(values[instruction[1]])))
                continue
            left = values[instruction[1]]
            right = values[instruction[2]]
            if opcode == "add":
                values.append(f32(left + right))
            elif opcode == "sub":
                values.append(f32(left - right))
            elif opcode == "mul":
                values.append(f32(left * right))
            elif opcode == "div":
                values.append(f32(_divide(left, right)))
            elif opcode == "min":
                values.append(right if left > right else left)
            elif opcode == "max":
                values.append(left if left > right else right)
            elif opcode == "eq":
                values.append(1.0 if left == right else 0.0)
            elif opcode == "gt":
                values.append(1.0 if left > right else 0.0)
            else:
                # binary64 holds the exact binary32 product, and 53 >= 2 * 24 + 2
                # makes the second rounding harmless, so this is fmaf exactly.
                values.append(f32(left * right + values[instruction[3]]))

        result = list(pixel)
        for channel, output in enumerate(self.outputs):
            if output is not None:
                result[channel] = _quantize(values[output.value], output.quantize)
        return tuple(result)  # type: ignore[return-value]


def _divide(left: float, right: float) -> float:
    if right == 0.0:
        if left == 0.0:
            return math.nan
        return math.copysign(
            math.inf, math.copysign(1.0, left) * math.copysign(1.0, right)
        )
    return left / right


def _quantize(value: float, rule: str) -> int:
    maximum = quantizer_maximum(rule)
    # "not (value > 0)" rather than "value <= 0" so a NaN takes this branch,
    # which is what the generated C does too.
    if not value > 0.0:
        return 0
    if value >= float(maximum):
        return maximum
    if rule.startswith("truncate_saturate"):
        return int(value)  # C's float-to-int conversion truncates toward zero.
    return round(value)  # lrintf under the default mode: nearest, ties to even.


#: Largest magnitude a binary32 can hold, above which a product overflows.
_FLOAT32_MAX = f32(3.4028234663852886e38)


def multiply_is_exact(multiplier: float, operand_bound: float) -> bool:
    """Whether ``x * multiplier`` is exact for every ``|x| <= operand_bound``.

    When it is, fusing the multiply into a following add cannot change the
    result: the fused form rounds ``x * multiplier + c`` once and the separate
    form rounds an already-exact product and then the sum, which is the same
    single rounding.  A lowerer can then transcribe upstream's ``a * b + c`` as
    a plain multiply and add and stay bit-exact whether or not the target
    contracts it -- which is why ``colorbalance``, whose multiply-adds all scale
    by ``4.f``, needs no fused instruction and no target knowledge at all.

    Scaling by a power of two moves the exponent and leaves the significand
    alone, so the only ways to lose a bit are overflowing to infinity or
    falling into the subnormals.  Scaling up cannot underflow, so this accepts
    only ``|multiplier| >= 1`` and requires the widest product to stay finite.
    """

    scale = f32(multiplier)
    if scale == 0.0:
        return True
    magnitude = abs(scale)
    if magnitude < 1.0 or math.frexp(magnitude)[0] != 0.5:
        return False
    return abs(f32(operand_bound)) * magnitude <= _FLOAT32_MAX


class ExprBuilder:
    """Emits instructions in evaluation order, sharing identical ones.

    Two instructions with the same opcode and operands compute the same value,
    so the builder returns the first one rather than appending a duplicate.
    That is what keeps a transcription of an upstream expression -- which names
    the same subexpression once per colour channel -- from tripling the
    generated C, and it cannot change a result.
    """

    def __init__(self) -> None:
        self._instructions: list[tuple[Any, ...]] = []
        self._index: dict[tuple[Any, ...], int] = {}

    def _emit(self, *instruction: Any) -> int:
        existing = self._index.get(instruction)
        if existing is not None:
            return existing
        index = len(self._instructions)
        self._instructions.append(instruction)
        self._index[instruction] = index
        return index

    def channel(self, channel: int) -> int:
        if not 0 <= channel <= 3:
            raise ExprError(f"channel {channel} is outside 0..3")
        return self._emit("channel", channel)

    def const(self, value: float) -> int:
        return self._emit("const", _hex_float32(value))

    def neg(self, value: int) -> int:
        return self._emit("neg", value)

    def abs(self, value: int) -> int:
        return self._emit("abs", value)

    def floor(self, value: int) -> int:
        return self._emit("floor", value)

    def lrintf(self, value: int) -> int:
        return self._emit("lrintf", value)

    def add(self, left: int, right: int) -> int:
        return self._emit("add", left, right)

    def sub(self, left: int, right: int) -> int:
        return self._emit("sub", left, right)

    def mul(self, left: int, right: int) -> int:
        return self._emit("mul", left, right)

    def div(self, left: int, right: int) -> int:
        return self._emit("div", left, right)

    def min(self, left: int, right: int) -> int:
        return self._emit("min", left, right)

    def max(self, left: int, right: int) -> int:
        return self._emit("max", left, right)

    def eq(self, left: int, right: int) -> int:
        return self._emit("eq", left, right)

    def gt(self, left: int, right: int) -> int:
        return self._emit("gt", left, right)

    def fma(self, left: int, right: int, addend: int) -> int:
        return self._emit("fma", left, right, addend)

    def multiply_add(self, left: int, right: int, addend: int, *, fused: bool) -> int:
        """Emit upstream's ``left * right + addend`` the way its compiler did."""

        if fused:
            return self.fma(left, right, addend)
        return self.add(self.mul(left, right), addend)

    def min3(self, first: int, second: int, third: int) -> int:
        return self.min(self.min(first, second), third)

    def max3(self, first: int, second: int, third: int) -> int:
        return self.max(self.max(first, second), third)

    def clipf(self, value: int, minimum: int, maximum: int) -> int:
        """``av_clipf``, which is ``FFMIN(FFMAX(a, amin), amax)``."""

        return self.min(self.max(value, minimum), maximum)

    def build(self, outputs: dict[int, tuple[int, str]]) -> ExprProgram:
        stored: list[Output | None] = [None, None, None, None]
        for channel, (value, quantize) in outputs.items():
            if not 0 <= channel <= 3:
                raise ExprError(f"channel {channel} is outside 0..3")
            if quantize not in QUANTIZERS:
                raise ExprError(f"unsupported output quantizer {quantize!r}")
            stored[channel] = Output(value, quantize)
        program = ExprProgram(tuple(self._instructions), tuple(stored))  # type: ignore[arg-type]
        # Round-trip through the validator so a builder mistake fails here
        # rather than inside a kernel.
        return ExprProgram.from_dict(program.as_dict())


def c_float_literal(value: float) -> str:
    """Render one binary32 constant as an exact C hexadecimal literal."""

    text = f32(value).hex()
    mantissa, _, exponent = text.partition("p")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}p{exponent}f"


#: Helpers the generated C needs when an 8-bit program is present.
C_HELPERS = (
    "static inline uint8_t lavfi_truncate_saturate_u8(float value)",
    "{",
    "    /* av_clip_uint8((int)value), clamped in float first: the same mapping",
    "     * for every input C defines, and defined for the ones it does not. */",
    "    if (!(value > 0.0f))",
    "        return 0;",
    "    if (value >= 255.0f)",
    "        return 255;",
    "    return (uint8_t)(int)value;",
    "}",
    "",
    "static inline uint8_t lavfi_lrintf_saturate_u8(float value)",
    "{",
    "    if (!(value > 0.0f))",
    "        return 0;",
    "    if (value >= 255.0f)",
    "        return 255;",
    "    return (uint8_t)lrintf(value);",
    "}",
)


def c_helpers(depth: int) -> tuple[str, ...]:
    """The quantizer helpers a program at *depth* calls.

    At eight bits this is :data:`C_HELPERS` verbatim, so a kernel generated
    before depths existed is still generated character for character.  Above
    it, the same two functions are emitted against ``av_clip_uintp2_c``'s bound
    and return a ``uint16_t``.
    """

    if depth == 8:
        return C_HELPERS
    maximum = (1 << depth) - 1
    return (
        f"static inline uint16_t lavfi_truncate_saturate_u{depth}(float value)",
        "{",
        f"    /* av_clip_uintp2_c((int)value, {depth}), clamped in float first: the",
        "     * same mapping for every input C defines, and defined for the",
        "     * ones it does not. */",
        "    if (!(value > 0.0f))",
        "        return 0;",
        f"    if (value >= {maximum}.0f)",
        f"        return {maximum};",
        "    return (uint16_t)(int)value;",
        "}",
        "",
        f"static inline uint16_t lavfi_lrintf_saturate_u{depth}(float value)",
        "{",
        "    if (!(value > 0.0f))",
        "        return 0;",
        f"    if (value >= {maximum}.0f)",
        f"        return {maximum};",
        "    return (uint16_t)lrintf(value);",
        "}",
    )

_C_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def render_c(
    program: ExprProgram, prefix: str, channels: tuple[int, ...], indent: str
) -> list[str]:
    """Emit one program as C, reading and writing the loop's ``c0``..``c3``.

    Only the channels this loop carries are stored back; a caller running one
    plane at a time has already been refused by ``validate_ir`` if the program
    reads across planes it cannot see together.
    """

    lines = [f"{indent}{{"]
    for index, instruction in enumerate(program.instructions):
        opcode = instruction[0]
        name = f"{prefix}_{index}"
        if opcode == "channel":
            expression = f"(float)c{instruction[1]}"
        elif opcode == "const":
            value = float.fromhex(instruction[1])
            expression = f"{c_float_literal(value)} /* {value:g} */"
        elif opcode == "neg":
            expression = f"-{prefix}_{instruction[1]}"
        elif opcode == "abs":
            expression = f"fabsf({prefix}_{instruction[1]})"
        elif opcode == "floor":
            expression = f"floorf({prefix}_{instruction[1]})"
        elif opcode == "lrintf":
            expression = f"(float)lrintf({prefix}_{instruction[1]})"
        elif opcode == "fma":
            expression = (
                f"fmaf({prefix}_{instruction[1]}, {prefix}_{instruction[2]}, "
                f"{prefix}_{instruction[3]})"
            )
        elif opcode in {"min", "max"}:
            left = f"{prefix}_{instruction[1]}"
            right = f"{prefix}_{instruction[2]}"
            # FFMIN and FFMAX verbatim, so a signed zero or a NaN picks the
            # same operand here as it does upstream.
            taken, otherwise = (right, left) if opcode == "min" else (left, right)
            expression = f"{left} > {right} ? {taken} : {otherwise}"
        elif opcode in {"eq", "gt"}:
            operator = "==" if opcode == "eq" else ">"
            expression = (
                f"(float)({prefix}_{instruction[1]} {operator} "
                f"{prefix}_{instruction[2]})"
            )
        else:
            expression = (
                f"{prefix}_{instruction[1]} {_C_BINARY[opcode]} "
                f"{prefix}_{instruction[2]}"
            )
        lines.append(f"{indent}    const float {name} = {expression};")
    for channel in channels:
        output = program.outputs[channel]
        if output is None:
            continue
        lines.append(
            f"{indent}    c{channel} = lavfi_{output.quantize}"
            f"({prefix}_{output.value});"
        )
    lines.append(f"{indent}}}")
    return lines
