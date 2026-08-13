from __future__ import annotations

import unittest

from lavfi_cc.frontend import require_ir
from lavfi_cc.interpreter import (
    InterpreterError,
    interpret_into,
    interpret_pixel,
    interpret_rgba8,
)
from lavfi_cc.ir import Operation, PixelIR


def graph(region: str) -> str:
    return f"format=rgba,{region},format=rgba"


class PixelInterpreterTests(unittest.TestCase):
    def test_negate_respects_components_and_alpha(self) -> None:
        ir = require_ir(graph("negate=components=r+b+a"))
        self.assertEqual(interpret_pixel(ir, (0, 64, 128, 255)), (255, 64, 127, 0))

    def test_every_stage_reads_the_previous_quantized_bytes(self) -> None:
        ir = require_ir(graph("lutrgb=r=val*0.5,lutrgb=r=val*0.5"))
        # 3 * .5 truncates to 1 at the first boundary, then to 0 at the second.
        self.assertEqual(interpret_pixel(ir, (3, 9, 10, 11)), (0, 9, 10, 11))

    def test_mixer_sums_independently_rounded_contributions(self) -> None:
        ir = require_ir(
            graph("colorchannelmixer=rr=0.5:rg=0.5:gg=1:bb=1:aa=1:pc=none")
        )
        # Both 0.5 contributions round to even zero before they are summed.
        self.assertEqual(interpret_pixel(ir, (1, 1, 0, 0))[0], 0)
        self.assertEqual(interpret_pixel(ir, (3, 3, 0, 0))[0], 4)

    def test_levels_uses_float32_and_saturates_after_truncation(self) -> None:
        ir = require_ir(
            graph(
                "colorlevels=rimin=0.5019607843137255:rimax=1:"
                "romin=0:romax=1:preserve=none"
            )
        )
        self.assertEqual(interpret_pixel(ir, (127, 1, 2, 3))[0], 0)
        self.assertEqual(interpret_pixel(ir, (128, 1, 2, 3))[0], 0)
        self.assertEqual(interpret_pixel(ir, (255, 1, 2, 3))[0], 255)


class FrameInterpreterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ir = require_ir(graph("negate=components=r+g+b+a"))

    def test_tightly_packed_frame(self) -> None:
        source = bytes((0, 1, 2, 3, 250, 251, 252, 253))
        self.assertEqual(
            interpret_rgba8(self.ir, source, 2, 1),
            bytes((255, 254, 253, 252, 5, 4, 3, 2)),
        )

    def test_padded_positive_source_and_negative_destination_strides(self) -> None:
        source = bytearray([0xEE] * 24)
        source[2:10] = bytes((0, 1, 2, 3, 4, 5, 6, 7))
        source[13:21] = bytes((8, 9, 10, 11, 12, 13, 14, 15))
        destination = bytearray([0xAA] * 24)

        interpret_into(
            self.ir,
            source,
            destination,
            2,
            2,
            source_stride=11,
            destination_stride=-10,
            source_offset=2,
            destination_offset=12,
        )

        self.assertEqual(
            destination[12:20], bytes((255, 254, 253, 252, 251, 250, 249, 248))
        )
        self.assertEqual(
            destination[2:10], bytes((247, 246, 245, 244, 243, 242, 241, 240))
        )
        self.assertEqual(destination[0:2], bytes((0xAA, 0xAA)))
        self.assertEqual(destination[10:12], bytes((0xAA, 0xAA)))
        self.assertEqual(destination[20:24], bytes((0xAA,) * 4))

    def test_zero_source_stride_reuses_the_first_row(self) -> None:
        destination = bytearray(8)
        interpret_into(
            self.ir,
            bytes((1, 2, 3, 4)),
            destination,
            1,
            2,
            source_stride=0,
        )
        self.assertEqual(
            destination,
            bytes((254, 253, 252, 251, 254, 253, 252, 251)),
        )

    def test_rejects_short_or_aliased_buffers(self) -> None:
        with self.assertRaisesRegex(InterpreterError, "expected 8"):
            interpret_rgba8(self.ir, b"short", 2, 1)
        shared = bytearray(8)
        with self.assertRaisesRegex(InterpreterError, "distinct"):
            interpret_into(self.ir, shared, shared, 2, 1)

    def test_rejects_a_layout_outside_the_buffer(self) -> None:
        with self.assertRaisesRegex(InterpreterError, "buffer is too small"):
            interpret_into(
                self.ir,
                bytes(16),
                bytearray(16),
                2,
                2,
                source_stride=12,
            )


class InterpreterValidationTests(unittest.TestCase):
    def test_rejects_unknown_ir_versions(self) -> None:
        valid = require_ir(graph("negate"))
        ir = PixelIR(valid.operations, ir_version=99)
        with self.assertRaisesRegex(InterpreterError, "unsupported IR version"):
            interpret_pixel(ir, (0, 0, 0, 0))

    def test_rejects_a_missing_quantization_boundary(self) -> None:
        identity = tuple(tuple(range(256)) for _ in range(4))
        ir = PixelIR(
            (
                Operation("load_rgba8", {}),
                Operation("lut8", {"tables": identity}),
                Operation("lut8", {"tables": identity}),
                Operation("store_rgba8", {}),
            )
        )
        with self.assertRaisesRegex(InterpreterError, "must be quantize_rgba8"):
            interpret_pixel(ir, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
