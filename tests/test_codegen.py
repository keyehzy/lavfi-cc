from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from lavfi_cc.codegen import generate_c
from lavfi_cc.frontend import require_ir
from lavfi_cc.interpreter import interpret_rgba8
from lavfi_cc.native import (
    KernelExecutionError,
    KernelLoadError,
    NativeKernel,
    compile_generated_c,
    library_suffix,
)
from lavfi_cc.passes import optimize_ir


def graph(region: str) -> str:
    return f"format=rgba,{region},format=rgba"


HAS_CLANG = shutil.which("clang") is not None


class OptimizationPassTests(unittest.TestCase):
    def test_identity_elimination_removes_each_supported_identity(self) -> None:
        ir = require_ir(
            graph("lutrgb,colorlevels=preserve=none,colorchannelmixer=pc=none")
        )
        result = optimize_ir(ir)
        self.assertEqual(len(result.ir.operations), 2)
        self.assertEqual(result.changes[0], ("identity_elimination", 3))
        self.assertEqual(
            interpret_rgba8(result.ir, bytes((1, 2, 3, 4)), 1, 1),
            bytes((1, 2, 3, 4)),
        )
        self.assertEqual(
            result.ir.metadata_effects, ("remove_color_dependent_side_data",)
        )

    def test_lut_composition_preserves_intermediate_byte_semantics(self) -> None:
        ir = require_ir(graph("lutrgb=r=val*0.5,lutrgb=r=val*0.5"))
        result = optimize_ir(ir)
        self.assertEqual(result.changes[1], ("lut_composition", 1))
        table = result.ir.operations[1].parameters["tables"][0]
        self.assertEqual(table[3], 0)
        self.assertEqual(table[255], 63)

    def test_composition_can_reveal_an_identity(self) -> None:
        ir = require_ir(
            graph(
                "negate=components=r+g+b+a,"
                "negate=components=r+g+b+a"
            )
        )
        result = optimize_ir(ir)
        self.assertEqual(len(result.ir.operations), 2)
        self.assertEqual(
            result.changes,
            (("identity_elimination", 1), ("lut_composition", 1)),
        )

    def test_passes_can_be_disabled_independently(self) -> None:
        ir = require_ir(graph("lutrgb,negate,lutrgb=r=val"))
        result = optimize_ir(
            ir, identity_elimination=False, lut_composition=False
        )
        self.assertEqual(result.ir.serialize(), ir.serialize())
        self.assertEqual(
            result.changes,
            (("identity_elimination", 0), ("lut_composition", 0)),
        )


class CGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_exports_the_checked_abi(self) -> None:
        ir = require_ir(
            graph(
                "negate,colorlevels=rimin=.1:preserve=none,"
                "colorchannelmixer=rr=.9:rg=.1:pc=none"
            )
        )
        first = generate_c(ir)
        second = generate_c(ir)
        self.assertEqual(first.source, second.source)
        self.assertIn(ir.plan_hash, first.source)
        self.assertIn("lavfi_compiled_kernel", first.source)
        self.assertIn("materialized levels_f32_fma", first.source)
        self.assertIn("for (int y = 0; y < height; ++y)", first.source)


@unittest.skipUnless(HAS_CLANG, "Week 4 native tests require Clang")
class NativeKernelTests(unittest.TestCase):
    def test_generated_kernels_match_the_interpreter(self) -> None:
        generator = random.Random(0xC04E)
        cases = (
            "negate=components=r+g+b+a",
            "lutrgb=r=val*1.5-2.75:g=negval:b='clip(val,17,239)':a=val/2",
            "colorlevels=rimin=.9:rimax=.1:romin=.8:romax=.2:preserve=none",
            "colorchannelmixer=rr=-2:rg=2:rb=.5:ra=-.5:"
            "gr=1.5:gg=-1.5:gb=.25:ga=.75:"
            "br=-.1:bg=.1:bb=1:ba=0:ar=2:ag=-2:ab=1:aa=.5:pc=none",
            "colorchannelmixer=rr=.5:rg=.5:gg=1:bb=1:aa=1:pc=none,"
            "colorchannelmixer=rr=1.5:gr=.5:gg=.5:bb=1:aa=1:pc=none",
            "negate,lutrgb=r=val*1.125-7,colorlevels=gimin=.1:gimax=.9:"
            "gomin=.8:gomax=.2:preserve=none,"
            "colorchannelmixer=rr=.9:rg=.1:gg=.8:gb=.2:bb=1:aa=1:pc=none",
        )
        widths = (1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65)
        for case_index, chain in enumerate(cases):
            width = widths[case_index * 2]
            height = 1 + case_index % 3
            source = bytes(
                generator.randrange(256) for _ in range(width * height * 4)
            )
            ir = require_ir(graph(chain))
            expected = interpret_rgba8(ir, source, width, height)
            with self.subTest(chain=chain), NativeKernel.compile(ir) as kernel:
                self.assertEqual(kernel.process_rgba8(source, width, height), expected)

    def test_native_layout_supports_padding_zero_and_negative_strides(self) -> None:
        ir = require_ir(graph("negate=components=r+g+b+a"))
        source = bytearray([0xEE] * 24)
        source[2:10] = bytes((0, 1, 2, 3, 4, 5, 6, 7))
        source[13:21] = bytes((8, 9, 10, 11, 12, 13, 14, 15))
        destination = bytearray([0xAA] * 24)
        with NativeKernel.compile(ir) as kernel:
            kernel.process_into(
                source,
                destination,
                2,
                2,
                source_stride=11,
                destination_stride=-10,
                source_offset=2,
                destination_offset=12,
            )
            repeated = bytearray(8)
            kernel.process_into(
                bytes((1, 2, 3, 4)),
                repeated,
                1,
                2,
                source_stride=0,
            )
        self.assertEqual(
            destination[12:20], bytes((255, 254, 253, 252, 251, 250, 249, 248))
        )
        self.assertEqual(
            destination[2:10], bytes((247, 246, 245, 244, 243, 242, 241, 240))
        )
        self.assertEqual(
            repeated, bytes((254, 253, 252, 251, 254, 253, 252, 251))
        )

    def test_native_layout_rejects_aliases_and_short_buffers(self) -> None:
        ir = require_ir(graph("negate"))
        with NativeKernel.compile(ir) as kernel:
            with self.assertRaisesRegex(KernelExecutionError, "expected 8"):
                kernel.process_rgba8(b"short", 2, 1)
            shared = bytearray(8)
            with self.assertRaisesRegex(KernelExecutionError, "distinct"):
                kernel.process_into(shared, shared, 2, 1)

    def test_loader_rejects_wrong_abi_and_plan_hash(self) -> None:
        ir = require_ir(graph("negate"))
        generated = generate_c(ir)
        mutations = (
            (
                "LAVFI_KERNEL_ABI_VERSION,",
                "99u,",
                "ABI version 99",
            ),
            (
                f'"{ir.plan_hash}"',
                '"not-the-expected-plan"',
                "does not match",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (needle, replacement, message) in enumerate(mutations):
                with self.subTest(message=message):
                    altered = replace(
                        generated, source=generated.source.replace(needle, replacement)
                    )
                    artifact = compile_generated_c(
                        altered, root / f"bad-{index}{library_suffix()}"
                    )
                    with self.assertRaisesRegex(KernelLoadError, message):
                        NativeKernel(artifact.library_path, ir.plan_hash)


if __name__ == "__main__":
    unittest.main()
