from __future__ import annotations

import json
import unittest

from lavfi_cc.expressions import build_lut
from lavfi_cc.frontend import analyze_filtergraph, require_ir


def graph(region: str, prefix: str = "") -> str:
    return f"{prefix}format=rgba,{region},format=yuv420p"


class ExpressionTests(unittest.TestCase):
    def test_lut_truncates_toward_zero_then_saturates(self) -> None:
        table = build_lut("val*1.5-2.75")
        self.assertEqual(table[0], 0)
        self.assertEqual(table[2], 0)  # 0.25 truncates to zero.
        self.assertEqual(table[3], 1)  # 1.75 truncates to one.
        self.assertEqual(table[255], 255)

    def test_lut_functions_and_variables(self) -> None:
        table = build_lut("clip(max(negval, 20), 0, 200)")
        self.assertEqual((table[0], table[55], table[255]), (200, 200, 20))

    def test_limited_range_reaches_clipval_and_negval(self) -> None:
        # lutyuv's luma range is 16..235, so clipval saturates at both ends and
        # negval is av_clip(16 + 235 - val, 16, 235) rather than 255 - val.
        luma = build_lut("clipval", (16, 235))
        self.assertEqual((luma[0], luma[16], luma[124], luma[235], luma[255]),
                         (16, 16, 124, 235, 235))
        negated = build_lut("negval", (16, 235))
        self.assertEqual((negated[0], negated[124], negated[255]), (235, 127, 16))
        # Chroma runs 16..240, so the same expression is a different table.
        chroma = build_lut("negval", (16, 240))
        self.assertEqual((chroma[0], chroma[124], chroma[255]), (240, 132, 16))
        self.assertNotEqual(negated, chroma)

    def test_minval_and_maxval_follow_the_range(self) -> None:
        self.assertEqual(build_lut("minval", (16, 235))[7], 16)
        self.assertEqual(build_lut("maxval", (16, 240))[7], 240)
        self.assertEqual(build_lut("maxval")[7], 255)


class IRTests(unittest.TestCase):
    def test_every_stage_has_an_explicit_quantization_boundary(self) -> None:
        ir = require_ir(graph("negate,lutrgb=r=val*1.08,colorlevels=preserve=none"))
        self.assertEqual(ir.operations[0].kind, "load_rgba8")
        self.assertEqual(ir.operations[-1].kind, "store_rgba8")
        self.assertEqual(
            [operation.kind for operation in ir.operations].count("quantize_rgba8"), 3
        )

    def test_semantically_equivalent_negate_has_identical_serialization(self) -> None:
        implicit = require_ir(graph("negate"))
        explicit = require_ir(graph("negate=components=b+r+g:negate_alpha=1"))
        self.assertEqual(implicit.serialize(), explicit.serialize())
        self.assertEqual(implicit.plan_hash, explicit.plan_hash)

    def test_source_locations_do_not_affect_cache_serialization(self) -> None:
        direct = require_ir(graph("negate"))
        prefixed = require_ir(graph("negate", prefix="hflip,"))
        self.assertEqual(direct.serialize(), prefixed.serialize())
        self.assertNotEqual(
            direct.operations[1].source.filter_index,
            prefixed.operations[1].source.filter_index,
        )

    def test_lut_removes_color_dependent_side_data(self) -> None:
        ir = require_ir(graph("lutrgb=r=val"))
        self.assertEqual(ir.metadata_effects, ("remove_color_dependent_side_data",))

    def test_colorlevels_uses_quantized_points_and_float32_coefficients(self) -> None:
        ir = require_ir(graph("colorlevels=rimin=0.1:rimax=0.9:preserve=none"))
        matrix = next(operation for operation in ir.operations if operation.kind == "matrix4x4")
        self.assertEqual(matrix.parameters["offsets"][0]["input"], 26)
        self.assertEqual(matrix.parameters["input_max"][0], 230)
        self.assertEqual(matrix.parameters["coefficients"][0][0], float(255 / 204).hex())

    def test_mixer_rounds_each_contribution_ties_even(self) -> None:
        ir = require_ir(graph("colorchannelmixer=rr=0.5:pc=none"))
        matrix = next(operation for operation in ir.operations if operation.kind == "matrix4x4")
        red_from_red = matrix.parameters["contribution_tables"][0][0]
        self.assertEqual(red_from_red[1], 0)
        self.assertEqual(red_from_red[3], 2)

    def test_serialization_is_canonical_json(self) -> None:
        ir = require_ir(graph("negate"))
        decoded = json.loads(ir.serialize())
        self.assertEqual(decoded["ir_version"], 3)
        self.assertEqual(decoded["layout"], "rgba")
        self.assertNotIn("source", ir.serialize().decode("ascii"))
        self.assertEqual(ir.serialize(), ir.serialize())


class EligibilityTests(unittest.TestCase):
    def assert_code(self, source: str, code: str) -> None:
        analysis = analyze_filtergraph(source)
        self.assertFalse(analysis.eligible)
        self.assertIn(code, [diagnostic.code for diagnostic in analysis.diagnostics])

    def test_requires_both_explicit_boundaries(self) -> None:
        # yuv410p is a real format the backend does not implement, so it can
        # only ever close a region, never open one.
        self.assert_code("negate,format=yuv410p", "missing_input_boundary")
        self.assert_code("format=rgba,negate", "missing_output_boundary")

    def test_reports_each_unsupported_filter_in_region(self) -> None:
        analysis = analyze_filtergraph(graph("scale=2:2,fps=30"))
        self.assertEqual(
            [diagnostic.filter_index for diagnostic in analysis.diagnostics], [1, 2]
        )
        self.assertTrue(all(item.code == "unsupported_filter" for item in analysis.diagnostics))

    def test_rejects_runtime_and_nonlocal_options(self) -> None:
        self.assert_code(graph("negate=enable='gte(t,1)'"), "runtime_option")
        self.assert_code(graph("colorlevels=rimin=-0.1"), "frame_global_extrema")
        self.assert_code(graph("colorlevels=rimin=0.1:rimax=0.101"), "degenerate_levels")
        self.assert_code(
            graph("colorlevels=rimin=1:rimax=0:romin=.9:romax=.1"),
            "target_sensitive_levels",
        )
        self.assert_code(graph("colorchannelmixer=pc=lum"), "unsupported_preserve")

    def test_rejects_nondeterministic_or_unsupported_lut_expression(self) -> None:
        self.assert_code(graph("lutrgb=r=random(0)"), "unsupported_expression")
        self.assert_code(graph("lutrgb=r=val/0"), "unsupported_expression")
        self.assert_code(graph("lutrgb=r=w+val"), "unsupported_expression")

    def test_allows_opaque_filters_outside_the_fused_region(self) -> None:
        analysis = analyze_filtergraph("scale=10:10," + graph("negate") + ",fps=24")
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        self.assertEqual(
            analysis.rewritten_filtergraph,
            "scale=10:10,format=rgba,fused=kernel=KERNEL_PATH:"
            "kernel_root=KERNEL_ROOT:plan_hash="
            + analysis.ir.plan_hash
            + ":remove_color_side_data=0,format=yuv420p,fps=24",
        )

    def test_rejects_ambiguous_multiple_regions(self) -> None:
        self.assert_code(
            "format=rgba,negate,format=yuv420p,format=rgba,lutrgb,format=yuv420p",
            "ambiguous_regions",
        )


if __name__ == "__main__":
    unittest.main()
