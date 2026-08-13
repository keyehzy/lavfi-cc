"""Small, semantics-preserving optimization passes over the pixel IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interpreter import validate_ir
from .ir import Operation, PixelIR


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
    if evaluation == "levels_f32_fma":
        return _identity_levels(operation)
    if evaluation == "sum_i32_terms_rounded_ties_even":
        return _identity_mixer(operation)
    return False


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
    )


def _remove_identities(
    stages: list[tuple[Operation, Operation]],
) -> tuple[list[tuple[Operation, Operation]], int]:
    kept = [stage for stage in stages if not _is_identity(stage[0])]
    return kept, len(stages) - len(kept)


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


def optimize_ir(
    ir: PixelIR,
    *,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> PassResult:
    """Run the independently switchable Week 4 optimization passes."""

    validate_ir(ir)
    stages = _stages(ir)
    changes: list[tuple[str, int]] = []

    identity_changes = 0
    if identity_elimination:
        stages, identity_changes = _remove_identities(stages)
    changes.append(("identity_elimination", identity_changes))

    composition_changes = 0
    if lut_composition:
        stages, composition_changes = _compose_luts(stages)
    changes.append(("lut_composition", composition_changes))

    # Composition can reveal an identity (for example, two full negations).
    if identity_elimination:
        stages, newly_removed = _remove_identities(stages)
        identity_changes += newly_removed
        changes[0] = ("identity_elimination", identity_changes)

    return PassResult(_with_stages(ir, stages), tuple(changes))
