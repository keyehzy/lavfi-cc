"""The cross-channel float32 expression, and the three RGB filters on it.

The layout matrix in :mod:`tests.test_layouts` checks these filters against the
pinned oracle in every accepted RGB format.  What is checked here is the
reasoning: that an expression really does couple channels in a way no table
could, that the sampling-group rule refuses it exactly where it must, which
option spellings are refused and why, and -- for ``colorcontrast`` -- that the
one host property its bytes depend on is a checked claim rather than a guess.
"""

from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from lavfi_cc import curves as curve_tables
from lavfi_cc.expr import (
    ExprBuilder,
    ExprError,
    ExprProgram,
    f32,
    multiply_is_exact,
)
from lavfi_cc.frontend import analyze_filtergraph, require_ir
from lavfi_cc.interpreter import InterpreterError, interpret_pixel, validate_ir
from lavfi_cc.ir import Operation, PixelIR
from lavfi_cc.passes import optimize_ir
from lavfi_cc.target import (
    FUSED_MULTIPLY_ADD_VARIABLE,
    UnknownTargetError,
    target_fuses_multiply_add,
)


ROOT = Path(__file__).resolve().parents[1]


def rgb(region: str, layout: str = "rgba") -> str:
    return f"format={layout},{region},format={layout}"


def transform(ir: PixelIR, kind: str) -> Operation:
    """The single transform of one kind, asserting there is exactly one."""

    matches = [operation for operation in ir.operations if operation.kind == kind]
    assert len(matches) == 1, [operation.kind for operation in ir.operations]
    return matches[0]


def program(ir: PixelIR) -> ExprProgram:
    return ExprProgram.from_dict(transform(ir, "expr_f32").parameters["program"])


def tables(ir: PixelIR) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(table) for table in transform(ir, "lut8").parameters["tables"])


class ExprProgramTests(unittest.TestCase):
    def build(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "instructions": [["channel", 0], ["const", "0x1p+1"], ["mul", 0, 1]],
            "outputs": [{"value": 2, "quantize": "lrintf_saturate_u8"}, None, None, None],
        }
        base.update(overrides)
        return base

    def test_a_well_formed_program_round_trips(self) -> None:
        parsed = ExprProgram.from_dict(self.build())
        self.assertEqual(parsed.as_dict(), self.build())
        self.assertEqual(parsed.channels_read, frozenset({0}))
        self.assertEqual(parsed.channels_written, frozenset({0}))
        self.assertEqual(parsed.evaluate((100, 1, 2, 3)), (200, 1, 2, 3))

    def test_a_forward_reference_is_refused(self) -> None:
        # Single assignment with no forward references is what makes one
        # evaluation order the only possible one.
        with self.assertRaises(ExprError):
            ExprProgram.from_dict(
                self.build(instructions=[["channel", 0], ["mul", 0, 2], ["const", "0x1p+1"]])
            )

    def test_malformed_programs_are_refused(self) -> None:
        for description, override in (
            ("unknown opcode", {"instructions": [["cube", 0]]}),
            ("wrong arity", {"instructions": [["channel", 0], ["add", 0]]}),
            ("channel out of range", {"instructions": [["channel", 7]]}),
            ("non-binary32 constant", {"instructions": [["const", "0x1.0000000001p+0"]]}),
            ("unknown quantizer", {"outputs": [{"value": 2, "quantize": "round"}, None, None, None]}),
            ("too few outputs", {"outputs": [None, None, None]}),
        ):
            with self.subTest(case=description):
                with self.assertRaises(ExprError):
                    ExprProgram.from_dict(self.build(**override))

    def test_fma_rounds_once_where_a_separate_multiply_rounds_twice(self) -> None:
        # The two forms have to be distinguishable, or nothing below means
        # anything: this is the triple the toolchain probe uses too.
        left = float.fromhex("0x1.000002p+0")
        right = float.fromhex("0x1.000004p+0")
        exact = Fraction(left) * Fraction(right) - 1
        self.assertNotEqual(
            f32(f32(left * right) - 1.0),
            f32(exact.numerator / exact.denominator),
        )

        builder = ExprBuilder()
        a = builder.const(left)
        b = builder.const(right)
        c = builder.const(-1.0)
        # The two forms differ by exactly one ulp of the result, so scaling the
        # difference up by 2^45 turns it into a byte.
        scale = builder.const(2.0**45)
        gap = builder.sub(builder.fma(a, b, c), builder.add(builder.mul(a, b), c))
        # The same comparison where the multiplier is a power of two, which is
        # colorbalance's whole argument in miniature.
        four = builder.const(4.0)
        exact = builder.sub(
            builder.fma(a, four, c), builder.add(builder.mul(a, four), c)
        )
        built = builder.build(
            {
                0: (builder.mul(gap, scale), "truncate_saturate_u8"),
                1: (builder.mul(exact, scale), "truncate_saturate_u8"),
            }
        )
        self.assertEqual(built.evaluate((0, 0, 0, 0))[:2], (1, 0))

    def test_min_and_max_reproduce_the_ffmpeg_macros(self) -> None:
        # FFMIN(a, b) is "a > b ? b : a" and FFMAX(a, b) is "a > b ? a : b", so
        # with a positive and a negative zero neither comparison is true and
        # the minimum comes out positive while the maximum comes out negative
        # -- the opposite of fminf and fmaxf, which both normalize to +0.
        builder = ExprBuilder()
        positive = builder.const(0.0)
        negative = builder.const(-0.0)
        one = builder.const(1.0)
        built = builder.build(
            {
                # 1 / +0.0 is +inf and 1 / -0.0 is -inf, which the quantizer
                # separates into 255 and 0.
                0: (
                    builder.div(one, builder.min(positive, negative)),
                    "truncate_saturate_u8",
                ),
                1: (
                    builder.div(one, builder.max(positive, negative)),
                    "truncate_saturate_u8",
                ),
            }
        )
        self.assertEqual(built.evaluate((0, 0, 0, 0))[:2], (255, 0))

    def test_the_builder_shares_identical_instructions(self) -> None:
        builder = ExprBuilder()
        first = builder.add(builder.channel(0), builder.channel(1))
        second = builder.add(builder.channel(0), builder.channel(1))
        self.assertEqual(first, second)
        self.assertEqual(len(builder.build({}).instructions), 3)

    def test_multiply_is_exact_only_for_a_power_of_two_that_cannot_overflow(self) -> None:
        # colorbalance's whole target-independence argument rests on this.
        self.assertTrue(multiply_is_exact(4.0, 2.0))
        self.assertTrue(multiply_is_exact(-8.0, 1e30))
        self.assertTrue(multiply_is_exact(0.0, 1e38))
        self.assertFalse(multiply_is_exact(3.0, 2.0))
        self.assertFalse(multiply_is_exact(0.5, 2.0))
        self.assertFalse(multiply_is_exact(4.0, 1e38))


class ExprSamplingGroupTests(unittest.TestCase):
    """An expression reads every channel, so it needs one common sample grid."""

    def build(self, layout: str) -> PixelIR:
        builder = ExprBuilder()
        total = builder.add(
            builder.add(builder.channel(0), builder.channel(1)), builder.channel(2)
        )
        built = builder.build({0: (total, "truncate_saturate_u8")})
        return PixelIR(
            (
                Operation("load_rgba8", {}),
                Operation("expr_f32", {"program": built.as_dict()}),
                Operation("quantize_rgba8", {"mode": "expression_outputs"}),
                Operation("store_rgba8", {}),
            ),
            layout=layout,
        )

    def test_admissible_wherever_every_channel_shares_a_resolution(self) -> None:
        for layout in ("rgba", "rgb24", "gbrp", "gbrap", "yuv444p"):
            with self.subTest(layout=layout):
                validate_ir(self.build(layout))

    def test_refused_on_a_subsampled_layout(self) -> None:
        # A yuv420p chroma sample covers four luma samples, so there is no
        # single pixel whose luma and chroma an expression could mix.
        for layout in ("yuv422p", "yuv420p"):
            with self.subTest(layout=layout):
                with self.assertRaises(InterpreterError) as caught:
                    validate_ir(self.build(layout))
                self.assertIn("different resolutions", str(caught.exception))

    def test_the_rule_reaches_ir_built_by_hand(self) -> None:
        with self.assertRaises(InterpreterError):
            optimize_ir(self.build("yuv420p"))

    def test_a_mismatched_quantizer_is_refused(self) -> None:
        ir = self.build("rgba")
        broken = PixelIR(
            (
                ir.operations[0],
                ir.operations[1],
                Operation("quantize_rgba8", {"mode": "lookup_u8"}),
                ir.operations[3],
            ),
            layout="rgba",
        )
        with self.assertRaises(InterpreterError):
            validate_ir(broken)


class ColorbalanceTests(unittest.TestCase):
    def test_the_lightness_term_couples_the_channels(self) -> None:
        # This is the property no lut8 and no matrix4x4 can express: changing
        # blue alone moves the red output, because both go through l.
        ir = require_ir(rgb("colorbalance=rs=.6:rm=-.4"))
        dark = interpret_pixel(ir, (120, 40, 0, 255))
        bright = interpret_pixel(ir, (120, 40, 255, 255))
        self.assertNotEqual(dark[0], bright[0])

    def test_defaults_pass_every_byte_through(self) -> None:
        # Upstream has no early-out here: it divides by 255, adds three zeroed
        # terms, and multiplies back. That round-trip is exact for every byte,
        # so the filter is a no-op in value even though it still runs.
        ir = require_ir(rgb("colorbalance=rs=0"))
        for value in range(256):
            self.assertEqual(
                interpret_pixel(ir, (value, value, value, value)),
                (value, value, value, value),
            )

    def test_the_program_contains_no_fused_multiply_add(self) -> None:
        # Every multiply-add in get_component scales by 4.f, which is exact, so
        # both evaluations agree and the lowering needs no target knowledge.
        built = program(require_ir(rgb("colorbalance=rs=.3:gm=-.2:bh=.5")))
        self.assertFalse(
            any(instruction[0] == "fma" for instruction in built.instructions)
        )

    def test_the_plan_is_the_same_on_either_kind_of_host(self) -> None:
        graph = rgb("colorbalance=rs=.3:gm=-.2:bh=.5")
        hashes = set()
        for setting in ("1", "0"):
            with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, setting):
                hashes.add(require_ir(graph).plan_hash)
        self.assertEqual(len(hashes), 1)

    def test_alpha_is_copied_rather_than_computed(self) -> None:
        built = program(require_ir(rgb("colorbalance=rs=.5")))
        self.assertEqual(built.channels_written, frozenset({0, 1, 2}))
        self.assertEqual(built.channels_read, frozenset({0, 1, 2}))

    def test_preserve_lightness_is_refused(self) -> None:
        # preservel's multiply-adds scale by a computed value rather than a
        # power of two, so whether the host fuses them would decide bytes.
        analysis = analyze_filtergraph(rgb("colorbalance=rs=.5:pl=1"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "unsupported_preserve")
        self.assertTrue(analyze_filtergraph(rgb("colorbalance=rs=.5:pl=0")).eligible)

    def test_out_of_range_options_are_refused(self) -> None:
        # Upstream's option range is [-1, 1] and av_opt_set fails outside it,
        # so the graph does not configure at all.
        analysis = analyze_filtergraph(rgb("colorbalance=rs=1.5"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "out_of_range")


class ColorcontrastTests(unittest.TestCase):
    def test_defaults_are_an_identity_the_optimizer_removes(self) -> None:
        # The slice loop is "y < slice_end && sum > FLT_EPSILON", and the three
        # weights default to zero, so upstream never touches the frame.
        ir = require_ir(rgb("colorcontrast=rc=.5"))
        self.assertTrue(program(ir).is_identity)
        optimized = optimize_ir(ir)
        self.assertEqual(
            [operation.kind for operation in optimized.ir.operations],
            ["load_rgba8", "store_rgba8"],
        )

    def test_a_weight_sum_at_the_epsilon_boundary_is_still_an_identity(self) -> None:
        # FLT_EPSILON itself does not pass "sum > FLT_EPSILON".
        epsilon = "1.1920929e-7"
        self.assertEqual(f32(float(epsilon)), 2.0**-23)
        self.assertTrue(program(require_ir(rgb(f"colorcontrast=rc=.5:gmw={epsilon}"))).is_identity)
        self.assertFalse(program(require_ir(rgb("colorcontrast=rc=.5:gmw=.001"))).is_identity)

    def test_every_multiply_add_upstream_fuses_is_stated_in_the_ir(self) -> None:
        graph = rgb("colorcontrast=rc=.4:gm=.25:by=-.15:rcw=.9:gmw=.7:byw=.3:pl=.35")
        with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, "1"):
            fused = program(require_ir(graph))
        with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, "0"):
            separate = program(require_ir(graph))
        # Nine axis shifts, six weighted-sum terms, three lightness blends.
        self.assertEqual(
            sum(1 for item in fused.instructions if item[0] == "fma"), 18
        )
        self.assertEqual(
            sum(1 for item in separate.instructions if item[0] == "fma"), 0
        )

    def test_the_two_kinds_of_host_never_share_a_kernel(self) -> None:
        graph = rgb("colorcontrast=rc=.4:rcw=.9:gmw=.7")
        hashes = []
        for setting in ("1", "0"):
            with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, setting):
                hashes.append(require_ir(graph).plan_hash)
        self.assertNotEqual(*hashes)

    def test_an_unparseable_override_is_refused(self) -> None:
        with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, "maybe"):
            analysis = analyze_filtergraph(rgb("colorcontrast=rc=.4:rcw=.9"))
            self.assertFalse(analysis.eligible)
            self.assertEqual(analysis.diagnostics[0].code, "unknown_target")

    def test_alpha_is_never_written(self) -> None:
        built = program(require_ir(rgb("colorcontrast=rc=.4:rcw=.9:gmw=.7")))
        self.assertEqual(built.channels_written, frozenset({0, 1, 2}))

    def test_out_of_range_options_are_refused(self) -> None:
        for option in ("rc=1.5", "rcw=-0.5", "pl=2"):
            with self.subTest(option=option):
                analysis = analyze_filtergraph(rgb(f"colorcontrast={option}"))
                self.assertFalse(analysis.eligible)
                self.assertEqual(analysis.diagnostics[0].code, "out_of_range")


class TargetFusionClaimTests(unittest.TestCase):
    """The machine table is a claim about the toolchain; this checks it.

    :mod:`lavfi_cc.target` decides from the machine name whether the host's C
    compiler contracts ``a * b + c``, because asking a compiler would mean the
    analysis-only scanner ran one.  Nothing else verifies that claim, so this
    does, by compiling the question and running it.
    """

    PROBE = """
#include <math.h>
#include <stdio.h>

int main(void)
{
    /* volatile so the operands reach code generation instead of being folded:
     * Clang's constant folder evaluates the written expression and would
     * report no fusion on a target that fuses. */
    volatile float source[3] = {0x1.000002p+0f, 0x1.000004p+0f, -1.0f};
    const float a = source[0], b = source[1], c = source[2];
    /* At -ffp-contract=on, which is Clang's default and what the pinned
     * oracle is built with, this contracts wherever the target can. */
    const float written = a * b + c;
    const float fused = fmaf(a, b, c);
    printf("%d\\n", written == fused);
    return 0;
}
"""

    def test_the_machine_table_matches_the_toolchain(self) -> None:
        clang = os.environ.get("LAVFI_CC_CLANG") or "clang"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "probe.c"
            binary = work / "probe"
            source.write_text(self.PROBE, encoding="utf-8")
            compiled = subprocess.run(
                [clang, "-O2", "-std=c11", str(source), "-o", str(binary), "-lm"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if compiled.returncode != 0:
                self.skipTest(f"probe did not compile: {compiled.stderr.decode()}")
            result = subprocess.run(
                [str(binary)], stdout=subprocess.PIPE, check=True
            )
        observed = result.stdout.decode().strip() == "1"
        try:
            claimed = target_fuses_multiply_add()
        except UnknownTargetError:
            self.skipTest("this machine is not in the table")
        self.assertEqual(
            claimed,
            observed,
            "lavfi_cc.target disagrees with the C compiler about multiply-add "
            "fusion, so colorcontrast would not match the pinned oracle",
        )


class CurvesTests(unittest.TestCase):
    def test_curves_is_channel_independent_and_needs_no_expression(self) -> None:
        # dst[x + r] = graph[R][src[x + r]] is a table lookup per channel, so
        # curves needs no IR extension at all.
        ir = require_ir(rgb("curves=r='0/0 0.5/0.4 1/1'"))
        self.assertEqual(
            [operation.kind for operation in ir.operations],
            ["load_rgba8", "lut8", "quantize_rgba8", "store_rgba8"],
        )

    def test_an_unset_component_and_alpha_are_identities(self) -> None:
        red, green, blue, alpha = tables(require_ir(rgb("curves=r='0/1 1/0'")))
        identity = tuple(range(256))
        self.assertEqual(green, identity)
        self.assertEqual(blue, identity)
        # NB_COMP is three: graph[3] is the master curve, not an alpha one.
        self.assertEqual(alpha, identity)
        self.assertEqual((red[0], red[255]), (255, 0))

    def test_the_master_curve_is_composed_on_top_of_each_component(self) -> None:
        component = tables(require_ir(rgb("curves=r='0/0 0.5/0.4 1/1'")))[0]
        master = tables(require_ir(rgb("curves=m='0/0 0.5/0.6 1/1'")))[0]
        both = tables(require_ir(rgb("curves=r='0/0 0.5/0.4 1/1':m='0/0 0.5/0.6 1/1'")))
        self.assertEqual(both[0], tuple(master[value] for value in component))
        # The master applies to every colour channel, including untouched ones.
        self.assertEqual(both[1], master)

    def test_all_fills_only_the_components_left_unset(self) -> None:
        red, green, blue, _ = tables(
            require_ir(rgb("curves=all='0/0 0.5/0.3 1/1':g='0/0 0.5/0.7 1/1'"))
        )
        self.assertEqual(red, blue)
        self.assertNotEqual(red, green)

    def test_a_preset_only_supplies_what_the_caller_did_not(self) -> None:
        preset = tables(require_ir(rgb("curves=preset=vintage")))
        overridden = tables(require_ir(rgb("curves=preset=vintage:r='0/0 1/1'")))
        self.assertEqual(preset[1], overridden[1])
        self.assertNotEqual(preset[0], overridden[0])
        self.assertEqual(overridden[0], tuple(range(256)))

    def test_a_preset_number_means_the_same_as_its_name(self) -> None:
        # The option is an int upstream, with the names as constants.
        self.assertEqual(
            tables(require_ir(rgb("curves=preset=negative"))),
            tables(require_ir(rgb("curves=preset=8"))),
        )

    def test_the_two_interpolators_disagree(self) -> None:
        points = "r='0/0 0.25/0.5 0.75/0.6 1/1'"
        natural = tables(require_ir(rgb(f"curves={points}:interp=natural")))
        pchip = tables(require_ir(rgb(f"curves={points}:interp=pchip")))
        self.assertNotEqual(natural[0], pchip[0])
        # PCHIP is monotonic between key points; the natural spline is not.
        self.assertEqual(list(pchip[0]), sorted(pchip[0]))

    def test_two_spellings_of_one_component_are_refused(self) -> None:
        # Upstream keeps whichever it parsed last, which is a spelling that
        # means something other than what it says.
        analysis = analyze_filtergraph(rgb("curves=r='0/0 1/1':red='0/1 1/0'"))
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "conflicting_options")

    def test_file_options_are_refused(self) -> None:
        for option in ("psfile=curves.acv", "plot=out.plt"):
            with self.subTest(option=option):
                analysis = analyze_filtergraph(rgb(f"curves={option}"))
                self.assertFalse(analysis.eligible)
                self.assertEqual(analysis.diagnostics[0].code, "runtime_option")

    def test_key_points_upstream_rejects_are_refused(self) -> None:
        for description, points in (
            ("outside the unit square", "0/0 1.5/1"),
            ("not increasing", "0.5/0 0.2/1"),
            ("too close together", "0.5/0 0.5001/1"),
            ("not a number", "0/0 half/1"),
        ):
            with self.subTest(case=description):
                analysis = analyze_filtergraph(rgb(f"curves=r='{points}'"))
                self.assertFalse(analysis.eligible)
                self.assertEqual(analysis.diagnostics[0].code, "unsupported_curve")

    def test_one_key_point_paints_a_single_value(self) -> None:
        red = tables(require_ir(rgb("curves=r='0.5/0.25'")))[0]
        self.assertEqual(set(red), {63})

    def test_a_curve_on_a_byte_boundary_follows_the_host(self) -> None:
        # y = 0.05 + 0.95x is exactly 160/255 at input 155, so the fused form
        # lands just below the boundary and the separate form just above, and
        # the truncating CLIP turns that into 159 or 160. Ordinary key points
        # do this; the two hosts genuinely disagree, upstream included, so the
        # table follows whichever host this is.
        points = curve_tables.parse_points("0/0.05 1/1", "red")
        fused = curve_tables.interpolate_natural(points, curve_tables.Arithmetic(True))
        separate = curve_tables.interpolate_natural(
            points, curve_tables.Arithmetic(False)
        )
        self.assertEqual((fused[155], separate[155]), (159, 160))
        for setting, expected in (("1", fused), ("0", separate)):
            with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, setting):
                self.assertEqual(
                    tables(require_ir(rgb("curves=r='0/0.05 1/1'")))[0],
                    tuple(expected),
                )

    def test_a_portable_curve_never_asks_about_the_host(self) -> None:
        # The two evaluations agree for almost every curve, and then the table
        # -- and so the plan hash -- is the same on either kind of host.
        graph = rgb("curves=r='0/0 0.5/0.4 1/1':g='0/0.1 1/0.9'")
        hashes = set()
        for setting in ("1", "0"):
            with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, setting):
                hashes.add(require_ir(graph).plan_hash)
        self.assertEqual(len(hashes), 1)

    def test_a_boundary_curve_gets_its_own_plan_on_each_host(self) -> None:
        graph = rgb("curves=r='0/0.05 1/1'")
        hashes = []
        for setting in ("1", "0"):
            with EnvironmentVariable(FUSED_MULTIPLY_ADD_VARIABLE, setting):
                hashes.append(require_ir(graph).plan_hash)
        self.assertNotEqual(*hashes)


class EnvironmentVariable:
    """Set one variable for the duration of a block, restoring it after."""

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        self.previous: str | None = None

    def __enter__(self) -> "EnvironmentVariable":
        self.previous = os.environ.get(self.name)
        os.environ[self.name] = self.value
        return self

    def __exit__(self, *_: object) -> None:
        if self.previous is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.previous


if __name__ == "__main__":
    unittest.main()
