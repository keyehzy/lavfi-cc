"""Small, semantics-preserving optimization passes over the pixel IR."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

from .interpreter import validate_ir
from .ir import Operation, PixelIR


LEVELS_EVALUATION = "levels_f32_fma"
MIXER_EVALUATION = "sum_i32_terms_rounded_ties_even"
FOLDED_MIXER_EVALUATION = "sum_i32_lut_terms"
MIXER_EVALUATIONS = frozenset((MIXER_EVALUATION, FOLDED_MIXER_EVALUATION))


@dataclass(frozen=True)
class PassResult:
    """Optimized IR together with deterministic per-pass change counts."""

    ir: PixelIR
    changes: tuple[tuple[str, int], ...]

    @property
    def changed(self) -> bool:
        return any(count for _, count in self.changes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "passes": [
                {"name": name, "changes": count} for name, count in self.changes
            ],
            "optimized_plan_hash": self.ir.plan_hash,
        }


def _identity_lut(operation: Operation) -> bool:
    tables = operation.parameters["tables"]
    identity = tuple(range(256))
    return all(tuple(table) == identity for table in tables)


def _identity_levels(operation: Operation) -> bool:
    parameters = operation.parameters
    return all(
        offset["input"] == offset["output"]
        and parameters["input_max"][channel]
        == parameters["output_max"][channel]
        for channel, offset in enumerate(parameters["offsets"])
    )


def _identity_mixer(operation: Operation) -> bool:
    tables = operation.parameters["contribution_tables"]
    identity = tuple(range(256))
    zero = (0,) * 256
    return all(
        tuple(tables[output][input_]) == (identity if output == input_ else zero)
        for output in range(4)
        for input_ in range(4)
    )


def _is_identity(operation: Operation) -> bool:
    if operation.kind == "lut8":
        return _identity_lut(operation)
    if operation.kind != "matrix4x4":
        return False
    evaluation = operation.parameters["evaluation"]
    if evaluation == LEVELS_EVALUATION:
        return _identity_levels(operation)
    if evaluation in MIXER_EVALUATIONS:
        return _identity_mixer(operation)
    return False


def _f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _u8(value: int) -> int:
    return max(0, min(255, value))


def materialize_levels_tables(
    operation: Operation,
) -> tuple[tuple[int, ...], ...]:
    """Materialize the exact byte mapping of a validated levels operation."""

    parameters = operation.parameters
    output: list[tuple[int, ...]] = []
    for channel in range(4):
        coefficient = _f32(float.fromhex(parameters["coefficients"][channel][channel]))
        input_offset = parameters["offsets"][channel]["input"]
        output_offset = parameters["offsets"][channel]["output"]
        output.append(
            tuple(
                _u8(
                    int(
                        _f32(
                            (value - input_offset) * coefficient + output_offset
                        )
                    )
                )
                for value in range(256)
            )
        )
    return tuple(output)


def _stages(ir: PixelIR) -> list[tuple[Operation, Operation]]:
    body = ir.operations[1:-1]
    return [(body[index], body[index + 1]) for index in range(0, len(body), 2)]


def _with_stages(
    ir: PixelIR, stages: list[tuple[Operation, Operation]]
) -> PixelIR:
    operations = [ir.operations[0]]
    for transform, quantize in stages:
        operations.extend((transform, quantize))
    operations.append(ir.operations[-1])
    return PixelIR(
        tuple(operations),
        metadata_effects=ir.metadata_effects,
        ir_version=ir.ir_version,
        pixel_format=ir.pixel_format,
        layout=ir.layout,
    )


def _remove_identities(
    stages: list[tuple[Operation, Operation]],
) -> tuple[list[tuple[Operation, Operation]], int]:
    kept = [stage for stage in stages if not _is_identity(stage[0])]
    return kept, len(stages) - len(kept)


def _materialize_levels(
    stages: list[tuple[Operation, Operation]],
) -> tuple[list[tuple[Operation, Operation]], int]:
    output: list[tuple[Operation, Operation]] = []
    changes = 0
    for transform, quantize in stages:
        if (
            transform.kind == "matrix4x4"
            and transform.parameters.get("evaluation") == LEVELS_EVALUATION
        ):
            transform = Operation(
                "lut8",
                {"tables": materialize_levels_tables(transform)},
                transform.source,
            )
            changes += 1
        output.append((transform, quantize))
    return output, changes


def _compose_luts(
    stages: list[tuple[Operation, Operation]],
) -> tuple[list[tuple[Operation, Operation]], int]:
    output: list[tuple[Operation, Operation]] = []
    compositions = 0
    index = 0
    while index < len(stages):
        transform, quantize = stages[index]
        if transform.kind != "lut8":
            output.append((transform, quantize))
            index += 1
            continue

        tables = tuple(tuple(table) for table in transform.parameters["tables"])
        final_transform = transform
        final_quantize = quantize
        index += 1
        while index < len(stages) and stages[index][0].kind == "lut8":
            next_transform, next_quantize = stages[index]
            next_tables = next_transform.parameters["tables"]
            tables = tuple(
                tuple(next_tables[channel][tables[channel][value]] for value in range(256))
                for channel in range(4)
            )
            final_transform = next_transform
            final_quantize = next_quantize
            compositions += 1
            index += 1
        output.append(
            (
                Operation("lut8", {"tables": tables}, final_transform.source),
                final_quantize,
            )
        )
    return output, compositions


def _fold_lut_into_mixer(lut: Operation, mixer: Operation) -> Operation:
    lut_tables = lut.parameters["tables"]
    mixer_tables = mixer.parameters["contribution_tables"]
    tables = tuple(
        tuple(
            tuple(
                mixer_tables[output_channel][input_channel][
                    lut_tables[input_channel][value]
                ]
                for value in range(256)
            )
            for input_channel in range(4)
        )
        for output_channel in range(4)
    )
    return Operation(
        "matrix4x4",
        {
            "evaluation": FOLDED_MIXER_EVALUATION,
            "offsets": [0, 0, 0, 0],
            "contribution_tables": tables,
        },
        mixer.source,
    )


def _fold_luts_into_mixers(
    stages: list[tuple[Operation, Operation]],
) -> tuple[list[tuple[Operation, Operation]], int]:
    output: list[tuple[Operation, Operation]] = []
    changes = 0
    index = 0
    while index < len(stages):
        transform, _quantize = stages[index]
        if index + 1 < len(stages):
            mixer, mixer_quantize = stages[index + 1]
            if (
                transform.kind == "lut8"
                and mixer.kind == "matrix4x4"
                and mixer.parameters.get("evaluation") in MIXER_EVALUATIONS
            ):
                output.append(
                    (_fold_lut_into_mixer(transform, mixer), mixer_quantize)
                )
                changes += 1
                index += 2
                continue
        output.append(stages[index])
        index += 1
    return output, changes


def optimize_ir(
    ir: PixelIR,
    *,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> PassResult:
    """Run the semantics-preserving pixel optimization passes."""

    validate_ir(ir)
    stages = _stages(ir)
    changes: list[tuple[str, int]] = []

    identity_changes = 0
    if identity_elimination:
        stages, identity_changes = _remove_identities(stages)
    changes.append(("identity_elimination", identity_changes))

    levels_changes = 0
    composition_changes = 0
    mixer_folding_changes = 0
    if lut_composition:
        stages, levels_changes = _materialize_levels(stages)
        stages, composition_changes = _compose_luts(stages)
        stages, mixer_folding_changes = _fold_luts_into_mixers(stages)
    changes.append(("levels_materialization", levels_changes))
    changes.append(("lut_composition", composition_changes))
    changes.append(("lut_mixer_folding", mixer_folding_changes))

    # Composition can reveal an identity (for example, two full negations).
    if identity_elimination:
        stages, newly_removed = _remove_identities(stages)
        identity_changes += newly_removed
        changes[0] = ("identity_elimination", identity_changes)

    optimized = _with_stages(ir, stages)
    validate_ir(optimized)
    return PassResult(optimized, tuple(changes))
