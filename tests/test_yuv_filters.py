"""Semantics of the YUV-native filters and the sampling-group rule they need.

The layout matrix in :mod:`tests.test_layouts` checks these filters against the
pinned oracle in every accepted format.  What is checked here is the reasoning
behind the lowering: which of upstream's code paths each option set selects,
which option spellings are refused and why, and the IR rule that lets ``hue``
mix Cb into Cr on a subsampled layout when a colour matrix still may not.
"""

from __future__ import annotations

import unittest

from lavfi_cc.expr import ExprBuilder
from lavfi_cc.frontend import analyze_filtergraph, require_ir
from lavfi_cc.interpreter import InterpreterError, interpret_pixel, validate_ir
from lavfi_cc.ir import Operation, PixelIR
from lavfi_cc.layouts import get_layout
from lavfi_cc.passes import optimize_ir


def yuv(region: str, layout: str = "yuv420p") -> str:
    return f"format={layout},{region},format={layout}"


def tables(ir: PixelIR) -> tuple[tuple[int, ...], ...]:
    """The single lut8 stage's four tables."""

    luts = [
        operation for operation in ir.operations if operation.kind == "lut8"
    ]
    assert len(luts) == 1, [operation.kind for operation in ir.operations]
    return tuple(tuple(table) for table in luts[0].parameters["tables"])


class LutyuvTests(unittest.TestCase):
    def test_components_use_the_limited_range(self) -> None:
        # negval is av_clip(minval + maxval - val, minval, maxval), and luma
        # and chroma have different maxima, so one expression is two tables.
        luma, u, v, _ = tables(require_ir(yuv("lutyuv=y=negval:u=negval:v=negval")))
        self.assertEqual((luma[0], luma[124], luma[255]), (235, 127, 16))
        self.assertEqual((u[0], u[124], u[255]), (240, 132, 16))
        self.assertEqual(u, v)
        self.assertNotEqual(luma, u)

    def test_default_expression_clips_rather_than_passes_through(self) -> None:
        # The default is clipval, not val: lutyuv on its own is not an identity
        # because it clamps into the limited range.
        luma, u, _, _ = tables(require_ir(yuv("lutyuv=y=clipval")))
        self.assertEqual((luma[0], luma[15], luma[16], luma[235], luma[255]),
                         (16, 16, 16, 235, 235))
        self.assertEqual((u[0], u[240], u[255]), (16, 240, 240))

    def test_alpha_expression_is_accepted_and_never_stored(self) -> None:
        # yuv420p carries no alpha plane, so upstream never even evaluates the
        # expression -- config_props builds one table per component the format
        # has -- and the channel is simply not written.
        analysis = analyze_filtergraph(yuv("lutyuv=y=negval:a=negval"))
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        self.assertEqual(get_layout("yuv420p").stored_channels, (0, 1, 2))

    def test_alpha_is_full_range_where_the_layout_carries_it(self) -> None:
        # On yuva420p the alpha table is applied. Its range is the one
        # config_props gives component A -- 0..255 -- rather than luma's
        # 16..235, so the same negval expression is two different tables.
        luma, _, _, alpha = tables(
            require_ir(yuv("lutyuv=y=negval:a=negval", "yuva420p"))
        )
        self.assertEqual((alpha[0], alpha[128], alpha[255]), (255, 127, 0))
        self.assertEqual((luma[0], luma[128], luma[255]), (235, 123, 16))
        self.assertEqual(get_layout("yuva420p").stored_channels, (0, 1, 2, 3))

    def test_the_default_alpha_expression_is_an_identity(self) -> None:
        # clipval over 0..255 clamps nothing, so lutyuv leaves alpha alone
        # unless the caller asks for it -- which is what upstream's copy of the
        # plane through its own table amounts to.
        self.assertEqual(
            tables(require_ir(yuv("lutyuv=y=negval", "yuva420p")))[3],
            tuple(range(256)),
        )

    def test_gamma_functions_are_refused(self) -> None:
        # gammaval and gammaval709 are vf_lut.c's own funcs1 entries, which
        # this compiler's expression subset does not evaluate.
        analysis = analyze_filtergraph(yuv("lutyuv=y='gammaval(0.5)'"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "unsupported_expression")


class EqTests(unittest.TestCase):
    def path_tables(self, region: str) -> tuple[tuple[int, ...], ...]:
        return tables(require_ir(yuv(region)))

    def test_all_defaults_copy_the_plane_rather_than_running_the_kernel(self) -> None:
        # check_values sends contrast=1, brightness=0, gamma=1 down the
        # plane-copy path. Running process_c with those same values would
        # subtract one from every sample, so the two are genuinely different
        # mappings and picking the wrong one is an off-by-one on every pixel.
        identity = tuple(range(256))
        for region in ("eq", "eq=contrast=1.0:brightness=0.0:gamma=1.0"):
            with self.subTest(region=region):
                self.assertEqual(self.path_tables(region)[0], identity)

    def test_process_path_is_integer_arithmetic(self) -> None:
        luma = self.path_tables("eq=brightness=0.1")[0]
        # ((v * 4096) >> 12) + brightness offset, saturating at both ends.
        self.assertEqual((luma[0], luma[1], luma[128]), (25, 26, 153))
        self.assertEqual(luma[255], 255)

    def test_contrast_bound_switches_to_the_gamma_lut(self) -> None:
        # process_c is used only while |contrast| < 7.9; beyond that upstream
        # builds the float table instead, and the two disagree.
        inside = self.path_tables("eq=contrast=7.89")[0]
        outside = self.path_tables("eq=contrast=7.9")[0]
        self.assertNotEqual(inside, outside)

    def test_saturation_drives_the_chroma_planes_only(self) -> None:
        luma, u, v, alpha = self.path_tables("eq=saturation=0.5")
        self.assertEqual(luma, tuple(range(256)))
        self.assertEqual(alpha, tuple(range(256)))
        self.assertEqual(u, v)
        self.assertNotEqual(u, luma)

    def test_out_of_range_values_are_clamped_not_refused(self) -> None:
        # av_clipf clamps, so FFmpeg accepts these graphs and so must we.
        for region, equivalent in (
            ("eq=contrast=-9999", "eq=contrast=-1000"),
            ("eq=brightness=7", "eq=brightness=1"),
            ("eq=gamma=99", "eq=gamma=10"),
            ("eq=saturation=-4", "eq=saturation=0"),
        ):
            with self.subTest(region=region):
                self.assertEqual(
                    require_ir(yuv(region)).plan_hash,
                    require_ir(yuv(equivalent)).plan_hash,
                )

    def test_per_frame_evaluation_is_refused(self) -> None:
        analysis = analyze_filtergraph(yuv("eq=eval=frame:contrast=1.4"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "runtime_option")

    def test_expressions_over_frame_variables_are_refused(self) -> None:
        for region in ("eq=contrast=n", "eq=brightness='0.1*t'", "eq=gamma=1+1"):
            with self.subTest(region=region):
                analysis = analyze_filtergraph(yuv(region))
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "dynamic_expression"
                )

    def test_options_are_rounded_to_binary32_like_av_clipf(self) -> None:
        # av_clipf's parameters are float, so the option is rounded to binary32
        # before anything uses it. These two literals are different doubles but
        # the same float, so they must produce the same kernel; an
        # implementation that kept them in binary64 would disagree.
        self.assertEqual(
            require_ir(yuv("eq=contrast=1.3")).plan_hash,
            require_ir(yuv("eq=contrast=1.2999999523162841796875")).plan_hash,
        )


class HueTests(unittest.TestCase):
    def rotation(self, region: str) -> Operation:
        operations = [
            operation
            for operation in require_ir(yuv(region)).operations
            if operation.kind == "chroma_rotate_i32"
        ]
        self.assertEqual(len(operations), 1)
        return operations[0]

    def test_degrees_and_radians_agree_on_the_same_angle(self) -> None:
        self.assertEqual(
            self.rotation("hue=h=90").parameters,
            self.rotation("hue=H=1.5707963267948966").parameters,
        )

    def test_coefficients_are_16_16_fixed_point(self) -> None:
        self.assertEqual(
            self.rotation("hue=h=0").parameters, {"cosine": 1 << 16, "sine": 0}
        )
        # Saturation scales the vector's norm, so it scales both coefficients.
        self.assertEqual(
            self.rotation("hue=h=0:s=1.5").parameters,
            {"cosine": 3 << 15, "sine": 0},
        )
        self.assertEqual(self.rotation("hue=h=90").parameters["sine"], 1 << 16)

    def test_conflicting_angle_options_are_refused(self) -> None:
        # Upstream fails to configure a graph naming both.
        analysis = analyze_filtergraph(yuv("hue=h=90:H=1.5"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "conflicting_options")

    def test_brightness_only_touches_luma(self) -> None:
        luma, u, v, _ = tables(require_ir(yuv("hue=b=0.4")))
        identity = tuple(range(256))
        self.assertEqual((u, v), (identity, identity))
        self.assertNotEqual(luma, identity)
        # i + b * 25.5, truncated toward zero, then clamped.
        self.assertEqual((luma[0], luma[1], luma[245]), (10, 11, 255))

    def test_zero_brightness_leaves_the_luma_table_an_identity(self) -> None:
        # Upstream copies the luma plane when brightness is zero; the table is
        # the identity there, so one lowering describes both paths.
        self.assertEqual(tables(require_ir(yuv("hue=h=45")))[0], tuple(range(256)))

    def test_a_no_op_hue_optimizes_away_entirely(self) -> None:
        result = optimize_ir(require_ir(yuv("hue=h=0:s=1:b=0")))
        self.assertEqual(
            [operation.kind for operation in result.ir.operations],
            ["load_rgba8", "store_rgba8"],
        )

    def test_rotation_matches_upstreams_integer_arithmetic(self) -> None:
        ir = require_ir(yuv("hue=h=90"))
        # cos is 0 and sin is 1<<16 here, so u' rounds -(v - 128) + 128 and v'
        # rounds (u - 128) + 128, both with the + (1 << 15) half step.
        self.assertEqual(interpret_pixel(ir, (17, 200, 60, 0))[1:3], (196, 200))
        self.assertEqual(interpret_pixel(ir, (17, 128, 128, 0))[1:3], (128, 128))
        # Luma passes through untouched at the default brightness.
        self.assertEqual(interpret_pixel(ir, (17, 200, 60, 0))[0], 17)

    def test_a_zero_angle_is_exact_rather_than_libm_sensitive(self) -> None:
        # At a zero angle the coefficient is a multiple of one sixteenth for
        # any saturation, so it lands on a rounding tie about one time in
        # eight. C fixes cos(0) at exactly 1 and sin(0) at exactly 0, so no
        # libm can disagree and the tie must not be refused as uncertain --
        # this saturation is one that produces exactly 617460.5.
        self.assertEqual(
            self.rotation("hue=s=9.4217").parameters,
            {"cosine": 617460, "sine": 0},
        )
        self.assertTrue(analyze_filtergraph(yuv("hue=s=9.4217:b=-8.7454")).eligible)

    def test_saturation_beyond_the_bound_is_clamped(self) -> None:
        self.assertEqual(
            self.rotation("hue=h=0:s=25").parameters,
            self.rotation("hue=h=0:s=10").parameters,
        )


class SamplingGroupTests(unittest.TestCase):
    """The rule that decides which channel mixes a layout can express."""

    def test_chroma_channels_share_a_sampling_group(self) -> None:
        self.assertEqual(get_layout("yuv420p").sampling_groups, ((0,), (1, 2)))
        self.assertEqual(get_layout("yuv422p").sampling_groups, ((0,), (1, 2)))
        # Nothing is subsampled here, so every channel is co-sited.
        self.assertEqual(get_layout("yuv444p").sampling_groups, ((0, 1, 2),))
        self.assertEqual(get_layout("rgba").sampling_groups, ((0, 1, 2, 3),))

    def test_yuva_alpha_joins_luma_rather_than_chroma(self) -> None:
        # yuva420p keeps alpha at the frame's resolution, so it is co-sited
        # with luma and not with the chroma it is stored after.
        self.assertEqual(get_layout("yuva420p").sampling_groups, ((0, 3), (1, 2)))

    def build(self, transform: Operation, mode: str, layout: str) -> PixelIR:
        return PixelIR(
            (
                Operation("load_rgba8", {}),
                transform,
                Operation("quantize_rgba8", {"mode": mode}),
                Operation("store_rgba8", {}),
            ),
            layout=layout,
        )

    def rotation(self) -> Operation:
        return Operation("chroma_rotate_i32", {"cosine": 1 << 15, "sine": 1 << 15})

    def mixer(self) -> Operation:
        identity = tuple(range(256))
        zero = (0,) * 256
        return Operation(
            "matrix4x4",
            {
                "evaluation": "sum_i32_lut_terms",
                "offsets": [0, 0, 0, 0],
                "contribution_tables": tuple(
                    tuple(identity if output == input_ else zero for input_ in range(4))
                    for output in range(4)
                ),
            },
        )

    def luma_alpha_expression(self) -> Operation:
        """An expression reading luma and alpha together and writing luma."""

        builder = ExprBuilder()
        mixed = builder.add(builder.channel(0), builder.channel(3))
        program = builder.build(
            {0: (builder.min(mixed, builder.const(255.0)), "truncate_saturate_u8")}
        )
        return Operation("expr_f32", {"program": program.as_dict()})

    def test_a_chroma_rotation_is_admissible_on_a_subsampled_layout(self) -> None:
        # Cb and Cr are sampled at the same positions, so one loop sees both.
        for layout in ("yuv444p", "yuv422p", "yuv420p",
                       "yuva444p", "yuva422p", "yuva420p"):
            with self.subTest(layout=layout):
                validate_ir(self.build(self.rotation(), "saturate_i32_to_u8", layout))

    def test_a_colour_matrix_is_still_refused_on_a_subsampled_layout(self) -> None:
        # Luma and chroma have no common sample, so nothing can mix them.
        for layout in ("yuv422p", "yuv420p", "yuva422p", "yuva420p"):
            with self.subTest(layout=layout):
                with self.assertRaises(InterpreterError) as caught:
                    validate_ir(
                        self.build(self.mixer(), "saturate_i32_to_u8", layout)
                    )
                self.assertIn("different resolutions", str(caught.exception))

    def test_luma_and_alpha_may_mix_where_chroma_still_may_not(self) -> None:
        # What decides this is the partition, not whether the layout subsamples
        # anything: yuva420p subsamples chroma, but luma and alpha are both
        # full resolution and so share a sample grid. No accepted filter writes
        # such a program today, so it is built by hand -- the rule admits it for
        # exactly the reason it admits hue's chroma rotation.
        for layout in ("yuva444p", "yuva422p", "yuva420p"):
            with self.subTest(layout=layout):
                validate_ir(
                    self.build(
                        self.luma_alpha_expression(), "expression_outputs", layout
                    )
                )
        # Reaching across the same layout's chroma is still refused, so this is
        # a property of which channels are mixed rather than of the layout.
        with self.assertRaises(InterpreterError):
            validate_ir(
                self.build(self.mixer(), "saturate_i32_to_u8", "yuva420p")
            )

    def test_the_rule_reaches_ir_built_by_hand(self) -> None:
        # Constructing the IR directly must not dodge the check.
        with self.assertRaises(InterpreterError):
            optimize_ir(self.build(self.mixer(), "saturate_i32_to_u8", "yuv420p"))

    def test_a_rotation_rejects_a_mismatched_quantizer(self) -> None:
        with self.assertRaises(InterpreterError):
            validate_ir(self.build(self.rotation(), "lookup_u8", "yuv420p"))

    def test_coefficients_outside_the_saturation_bound_are_refused(self) -> None:
        wild = Operation("chroma_rotate_i32", {"cosine": 1 << 24, "sine": 0})
        with self.assertRaises(InterpreterError):
            validate_ir(self.build(wild, "saturate_i32_to_u8", "yuv420p"))


if __name__ == "__main__":
    unittest.main()
