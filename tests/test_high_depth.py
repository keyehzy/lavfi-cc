"""Sample depths above eight, and what changes with them.

The layout matrix in :mod:`tests.test_layouts` checks every accepted filter
against the pinned oracle in every accepted deep format.  What is checked here
is the reasoning: which formats each filter really advertises at which depth,
where upstream's constants stop being the eight-bit ones, what the kernel's
domain is and why a table is indexed through a clamp, and that adding all of
this left the eight-bit IR alone.
"""

from __future__ import annotations

import unittest

from lavfi_cc.codegen import generate_c
from lavfi_cc.expr import QUANTIZERS, quantizer_maximum
from lavfi_cc.expressions import build_lut
from lavfi_cc.filters import filter_supports_format
from lavfi_cc.frontend import analyze_filtergraph, require_ir
from lavfi_cc.interpreter import (
    InterpreterError,
    interpret_pixel,
    interpret_rgba8,
    validate_ir,
)
from lavfi_cc.ir import Operation, PixelIR
from lavfi_cc.layouts import HIGH_DEPTH_LAYOUTS, LAYOUTS, get_layout


def graph(layout: str, chain: str) -> str:
    return f"format={layout},{chain},format={layout}"


class LayoutTableTests(unittest.TestCase):
    def test_a_sample_is_one_byte_up_to_eight_bits_and_two_above(self) -> None:
        for name, layout in LAYOUTS.items():
            with self.subTest(layout=name):
                self.assertEqual(layout.sample_bytes, 1 if layout.depth <= 8 else 2)
                self.assertEqual(layout.high_depth, layout.depth > 8)
                self.assertEqual(layout.max_value, (1 << layout.depth) - 1)

    def test_a_row_is_twice_as_many_bytes_as_samples_above_eight_bits(self) -> None:
        # The step and the offsets count samples, so only the byte sizing moves
        # with the depth; that is what keeps one addressing scheme for both.
        self.assertEqual(get_layout("yuv420p").plane_row_bytes(0, 17), 17)
        self.assertEqual(get_layout("yuv420p10le").plane_row_bytes(0, 17), 34)
        self.assertEqual(get_layout("yuv420p10le").step, 1)
        # rgb48le is three samples of two bytes rather than six of one.
        self.assertEqual(get_layout("rgb48le").step, 3)
        self.assertEqual(get_layout("rgb48le").plane_row_bytes(0, 4), 24)
        self.assertEqual(get_layout("rgba64le").frame_size(4, 2), 64)

    def test_a_deep_layout_keeps_its_familys_plane_order(self) -> None:
        self.assertEqual(get_layout("gbrp10le").planes, (2, 0, 1, None))
        self.assertEqual(get_layout("gbrap16le").planes, (2, 0, 1, 3))
        self.assertEqual(get_layout("yuv420p10le").planes, (0, 1, 2, None))
        self.assertEqual(get_layout("yuva420p16le").sampling_groups, ((0, 3), (1, 2)))
        self.assertEqual(get_layout("bgr48le").offsets, (2, 1, 0, None))

    def test_every_layout_has_its_own_abi_identifier(self) -> None:
        identifiers = [layout.abi_id for layout in LAYOUTS.values()]
        self.assertEqual(len(set(identifiers)), len(identifiers))


class AdvertisedFormatTests(unittest.TestCase):
    """The filter subset narrows with depth, and not uniformly."""

    def test_eq_is_the_one_filter_with_no_deep_format_at_all(self) -> None:
        # vf_eq.c lists 8-bit formats only, so a deep YUV island can never
        # contain it however the rest of the chain is written.
        self.assertTrue(filter_supports_format("eq", "yuv420p"))
        for name in HIGH_DEPTH_LAYOUTS:
            with self.subTest(layout=name):
                self.assertFalse(filter_supports_format("eq", name))
        analysis = analyze_filtergraph(
            graph("yuv420p10le", "eq=contrast=1.1,negate")
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "format_not_advertised")

    def test_hue_advertises_ten_bits_and_nothing_else_above_eight(self) -> None:
        for name in ("yuv420p10le", "yuv444p10le", "yuva420p10le"):
            self.assertTrue(filter_supports_format("hue", name), name)
        for name in ("yuv420p9le", "yuv420p12le", "yuv420p16le", "yuva420p16le"):
            self.assertFalse(filter_supports_format("hue", name), name)

    def test_lutyuv_and_hue_disagree_about_the_deep_yuva_formats(self) -> None:
        # vf_lut.c lists the alpha-bearing planar YUV formats at sixteen bits
        # only; vf_hue.c lists them at ten only. So the two never share a deep
        # yuva format, and a run containing both is not contiguous in one.
        self.assertTrue(filter_supports_format("lutyuv", "yuva420p16le"))
        self.assertFalse(filter_supports_format("lutyuv", "yuva420p10le"))
        self.assertTrue(filter_supports_format("hue", "yuva420p10le"))
        self.assertFalse(filter_supports_format("hue", "yuva420p16le"))
        analysis = analyze_filtergraph(
            graph("yuva420p10le", "lutyuv=y=negval,hue=s=1.1")
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "format_not_advertised")
        # hue alone is fine there, and so is lutyuv one depth up.
        self.assertTrue(
            analyze_filtergraph(graph("yuva420p10le", "hue=s=1.1,hue=h=4")).eligible
        )
        self.assertTrue(
            analyze_filtergraph(
                graph("yuva420p16le", "lutyuv=y=negval,negate")
            ).eligible
        )

    def test_lutrgb_carries_the_rgb_orders_and_not_the_bgr_ones(self) -> None:
        # vf_lut.c's RGB list has RGB48LE and RGBA64LE but neither BGR order,
        # while vf_negate.c has its own list and carries all four.
        for name in ("rgb48le", "rgba64le"):
            self.assertTrue(filter_supports_format("lutrgb", name), name)
        for name in ("bgr48le", "bgra64le"):
            self.assertFalse(filter_supports_format("lutrgb", name), name)
            self.assertTrue(filter_supports_format("negate", name), name)
            self.assertTrue(filter_supports_format("curves", name), name)
        analysis = analyze_filtergraph(
            graph("bgra64le", "lutrgb=r=val*1.1,negate")
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "format_not_advertised")


class UpstreamConstantTests(unittest.TestCase):
    """Where upstream's own numbers stop being the eight-bit ones."""

    def test_lutyuv_scales_the_limited_range_with_the_depth(self) -> None:
        # config_props gives luma 16..235 and chroma 16..240 shifted left by
        # depth - 8, and alpha the full range, which is also the bound every
        # component is finally clipped to.
        table = build_lut("negval", (64, 940), 10, 1023)
        self.assertEqual(len(table), 1024)
        self.assertEqual(table[0], 940)
        self.assertEqual(table[500], 64 + 940 - 500)
        self.assertEqual(table[1023], 64)
        ir = require_ir(graph("yuv420p10le", "lutyuv=y=negval"))
        luma = ir.operations[1].parameters["tables"][0]
        self.assertEqual(tuple(luma), table)

    def test_deep_planar_rgb_runs_to_255_shifted_rather_than_full_scale(self) -> None:
        # gbrp10le falls into config_props' default arm, whose maximum is
        # 255 << (depth - 8) = 1020 rather than 1023. Everything, including the
        # final clip, uses that.
        ir = require_ir(graph("gbrp10le", "lutrgb=r=negval"))
        red = ir.operations[1].parameters["tables"][0]
        self.assertEqual(red[0], 1020)
        self.assertEqual(red[1020], 0)
        self.assertEqual(red[1023], 0)
        # rgba64le is the one deep RGB arm that really is full range.
        packed = require_ir(graph("rgba64le", "lutrgb=r=negval"))
        self.assertEqual(packed.operations[1].parameters["tables"][0][0], 65535)

    def test_negate_is_max_minus_sample_at_every_depth(self) -> None:
        # vf_negate.c is its own filter rather than a vf_lut.c entry point, so
        # the limited luma range plays no part: it is (1 << depth) - 1 minus
        # the sample even on yuv420p10le.
        ir = require_ir(graph("yuv420p10le", "negate"))
        luma = ir.operations[1].parameters["tables"][0]
        self.assertEqual(luma[0], 1023)
        self.assertEqual(luma[1023], 0)
        self.assertEqual(luma[64], 959)

    def test_hue_scales_brightness_by_the_count_not_the_maximum(self) -> None:
        # create_luma_lut multiplies by 25.5 at eight bits and 102.4 at ten;
        # 255/10 and 1024/10 are not the same rule, and both are transcribed.
        deep = require_ir(graph("yuv420p10le", "hue=b=0.5"))
        self.assertEqual(deep.operations[1].parameters["tables"][0][0], int(51.2))
        shallow = require_ir(graph("yuv420p", "hue=b=0.5"))
        self.assertEqual(shallow.operations[1].parameters["tables"][0][0], int(12.75))

    def test_hue_rotates_chroma_about_the_depths_own_centre(self) -> None:
        deep = require_ir(graph("yuv420p10le", "hue=h=45"))
        rotate = deep.operations[3]
        self.assertEqual(rotate.kind, "chroma_rotate_i32")
        self.assertEqual(rotate.parameters["depth"], 10)
        # An unrotated chroma pair comes back unchanged about centre 512.
        self.assertEqual(interpret_pixel(require_ir(graph("yuv420p10le", "hue=s=1")),
                                         (100, 512, 700, 0))[1:3], (512, 700))

    def test_colorlevels_scales_its_endpoints_by_uint16_max_at_any_depth(self) -> None:
        # A quirk of vf_colorlevels.c: the option's [0, 1] endpoint is scaled
        # by UINT16_MAX whenever a sample is two bytes, so a gbrp10le endpoint
        # runs to 65535 while the samples it is compared against stop at 1023.
        ir = require_ir(graph("gbrp10le", "colorlevels=rimin=0.05"))
        levels = ir.operations[1]
        self.assertEqual(levels.parameters["offsets"][0]["input"], round(0.05 * 65535))
        self.assertEqual(levels.parameters["input_max"][0], 65535)
        shallow = require_ir(graph("gbrp", "colorlevels=rimin=0.05"))
        self.assertEqual(
            shallow.operations[1].parameters["offsets"][0]["input"], round(0.05 * 255)
        )


class KernelDomainTests(unittest.TestCase):
    """Samples the format cannot hold, and what a kernel does with them."""

    def test_a_table_covers_the_formats_own_domain(self) -> None:
        for name, entries in (
            ("yuv420p", 256),
            ("yuv420p9le", 512),
            ("yuv420p10le", 1024),
            ("yuv420p12le", 4096),
            ("yuv420p16le", 65536),
        ):
            with self.subTest(layout=name):
                ir = require_ir(graph(name, "negate"))
                self.assertEqual(len(ir.operations[1].parameters["tables"][0]), entries)

    def test_a_stray_sample_is_clamped_rather_than_read_past_the_table(self) -> None:
        # Upstream has no one answer here -- vf_lut.c carries a full 65536-entry
        # table, vf_hue.c clamps, and vf_curves.c reads past the end of its own
        # -- so the kernel defines the clamp and says so.
        ir = require_ir(graph("yuv420p10le", "negate"))
        self.assertEqual(interpret_pixel(ir, (1023, 0, 0, 0))[0], 0)
        with self.assertRaises(InterpreterError):
            interpret_pixel(ir, (2000, 0, 0, 0))
        # The generated C clamps rather than trusting the sample.
        source = generate_c(ir).source
        self.assertIn("c0 > 1023 ? 1023 : c0", source)

    def test_eight_and_sixteen_bit_tables_need_no_clamp_at_all(self) -> None:
        # A table there covers every value the sample type can hold, so the
        # subscript is the sample itself and the eight-bit text is unchanged.
        for name in ("yuv420p", "yuv420p16le"):
            with self.subTest(layout=name):
                source = generate_c(require_ir(graph(name, "negate"))).source
                self.assertIn("[c0];", source)
                self.assertNotIn("? ", source.split("static void")[1])


class GeneratedCodeTests(unittest.TestCase):
    def test_a_deep_kernel_loads_and_stores_sixteen_bit_samples(self) -> None:
        source = generate_c(require_ir(graph("yuv420p10le", "negate"))).source
        self.assertIn("const uint16_t *source_row = (const uint16_t *)(", source)
        self.assertIn("uint16_t c0 = source_row[x];", source)
        self.assertIn("static const uint16_t lut_0[4][1024]", source)

    def test_a_deep_kernel_refuses_to_compile_on_a_big_endian_host(self) -> None:
        # Every layout in the table is little-endian and the kernel loads a
        # native word, so the two only agree on a little-endian host.
        source = generate_c(require_ir(graph("gbrp10le", "negate"))).source
        self.assertIn("__ORDER_LITTLE_ENDIAN__", source)
        self.assertNotIn(
            "__ORDER_LITTLE_ENDIAN__",
            generate_c(require_ir(graph("gbrp", "negate"))).source,
        )

    def test_the_mixer_sums_deep_terms_in_a_wider_vector(self) -> None:
        # A term is a sample scaled by at most two, which fits an int16_t at
        # eight bits and does not above.
        deep = generate_c(
            require_ir(graph("gbrp16le", "colorchannelmixer=rr=0.9:rg=0.1"))
        ).source
        self.assertIn("typedef int32_t lavfi_i32x4", deep)
        shallow = generate_c(
            require_ir(graph("gbrp", "colorchannelmixer=rr=0.9:rg=0.1"))
        ).source
        self.assertIn("typedef int16_t lavfi_i16x4", shallow)

    def test_an_expression_quantizes_to_the_layouts_own_width(self) -> None:
        source = generate_c(
            require_ir(graph("gbrp10le", "colorbalance=rs=0.3"))
        ).source
        self.assertIn("static inline uint16_t lavfi_lrintf_saturate_u10", source)
        self.assertIn("if (value >= 1023.0f)", source)


class IRShapeTests(unittest.TestCase):
    """The eight-bit IR is exactly what it was; a deeper one is parallel."""

    def test_the_eight_bit_operations_are_unchanged(self) -> None:
        ir = require_ir(graph("rgba", "negate"))
        self.assertEqual(ir.pixel_format, "rgba8")
        self.assertEqual(
            [operation.kind for operation in ir.operations],
            ["load_rgba8", "lut8", "quantize_rgba8", "store_rgba8"],
        )
        self.assertEqual(set(ir.operations[1].parameters), {"tables"})

    def test_a_deeper_program_uses_the_parallel_operations(self) -> None:
        ir = require_ir(graph("rgba64le", "negate"))
        self.assertEqual(ir.pixel_format, "rgba16")
        self.assertEqual(
            [operation.kind for operation in ir.operations],
            ["load_rgba16", "lut16", "quantize_rgba16", "store_rgba16"],
        )
        self.assertEqual(ir.operations[1].parameters["depth"], 16)
        self.assertEqual(ir.operations[2].parameters["mode"], "lookup_u16")

    def test_a_program_may_not_declare_a_depth_the_layout_does_not_store(
        self,
    ) -> None:
        ir = require_ir(graph("yuv420p10le", "negate"))
        wrong = PixelIR(
            (
                ir.operations[0],
                Operation(
                    "lut16",
                    {**ir.operations[1].parameters, "depth": 12},
                ),
                ir.operations[2],
                ir.operations[3],
            ),
            pixel_format="rgba10",
            layout="yuv420p10le",
        )
        with self.assertRaises(InterpreterError):
            validate_ir(wrong)

    def test_a_pixel_format_that_contradicts_the_layout_is_refused(self) -> None:
        ir = require_ir(graph("yuv420p10le", "negate"))
        with self.assertRaises(InterpreterError):
            validate_ir(
                PixelIR(ir.operations, pixel_format="rgba8", layout="yuv420p10le")
            )

    def test_every_layout_still_gets_its_own_plan_hash(self) -> None:
        hashes = {
            name: analyze_filtergraph(graph(name, "negate")).ir.plan_hash
            for name in LAYOUTS
        }
        self.assertEqual(len(set(hashes.values())), len(LAYOUTS))

    def test_the_quantizer_names_carry_their_own_width(self) -> None:
        self.assertIn("truncate_saturate_u8", QUANTIZERS)
        self.assertIn("lrintf_saturate_u10", QUANTIZERS)
        self.assertEqual(quantizer_maximum("truncate_saturate_u8"), 255)
        self.assertEqual(quantizer_maximum("lrintf_saturate_u12"), 4095)


class FrameWalkTests(unittest.TestCase):
    def test_a_deep_frame_round_trips_through_the_interpreter(self) -> None:
        # Two negates compose to an identity at any depth, which is also what
        # exercises the little-endian pair the walkers read and write.
        for name in ("yuv420p10le", "gbrap16le", "rgb48le", "yuva422p16le"):
            with self.subTest(layout=name):
                layout = get_layout(name)
                width, height = 5, 3
                source = bytes(
                    range(1, layout.frame_size(width, height) + 1)
                )
                # Keep every sample inside the format's own domain.
                samples = [
                    int.from_bytes(source[index : index + 2], "little")
                    & layout.max_value
                    for index in range(0, len(source), 2)
                ]
                frame = b"".join(
                    value.to_bytes(2, "little") for value in samples
                )
                ir = require_ir(graph(name, "negate,negate"))
                self.assertEqual(interpret_rgba8(ir, frame, width, height), frame)


if __name__ == "__main__":
    unittest.main()
