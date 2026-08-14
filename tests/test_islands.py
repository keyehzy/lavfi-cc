from __future__ import annotations

import unittest

from lavfi_cc.frontend import analyze_filtergraph
from lavfi_cc.islands import (
    NATIVE,
    NEGOTIATED,
    scan_chain,
    select_islands,
)
from lavfi_cc.parser import parse_filtergraph, parse_filtergraph_script


def chain(source: str):
    return parse_filtergraph(source).filters


class IslandDiscoveryTests(unittest.TestCase):
    def test_finds_a_maximal_run_without_explicit_boundaries(self) -> None:
        scan = scan_chain(chain("negate,lutrgb=r=val*2,negate"), entry_format="rgba")
        self.assertEqual(len(scan.islands), 1)
        island = scan.islands[0]
        self.assertEqual((island.start, island.end), (0, 3))
        self.assertEqual(island.names, ("negate", "lutrgb", "negate"))
        self.assertTrue(island.fusible)
        self.assertEqual(island.eliminated_passes, 2)

    def test_unsupported_filter_splits_a_run(self) -> None:
        scan = scan_chain(
            chain("negate,lutrgb=r=val*2,scale=2:2,negate,negate"),
            entry_format="rgba",
        )
        self.assertEqual([(i.start, i.end) for i in scan.islands], [(0, 2), (3, 5)])
        self.assertEqual([r.name for r in scan.rejections], ["scale"])

    def test_scale_clears_the_pinned_format_but_crop_keeps_it(self) -> None:
        after_scale = scan_chain(
            chain("negate,scale=2:2,negate,negate"), entry_format="rgba"
        )
        self.assertEqual(after_scale.islands[1].boundary, NEGOTIATED)
        after_crop = scan_chain(
            chain("negate,crop=2:2,negate,negate"), entry_format="rgba"
        )
        self.assertEqual(after_crop.islands[1].boundary, NATIVE)

    def test_format_filter_pins_and_repins_the_working_format(self) -> None:
        scan = scan_chain(
            chain("format=yuv410p,negate,negate,format=rgba,negate,negate")
        )
        self.assertEqual(
            [island.boundary for island in scan.islands],
            [NATIVE, NATIVE],
        )
        self.assertEqual(scan.islands[0].working_format, "yuv410p")

    def test_a_supported_yuv_format_pins_a_native_island(self) -> None:
        scan = scan_chain(chain("format=yuv420p,negate,negate"))
        self.assertEqual([island.boundary for island in scan.islands], [NATIVE])
        self.assertEqual(scan.islands[0].working_format, "yuv420p")

    def test_a_rejected_option_splits_the_run_but_keeps_the_format(self) -> None:
        scan = scan_chain(
            chain("negate,colorlevels=rimin=-0.5,negate,negate"), entry_format="rgba"
        )
        self.assertEqual([(i.start, i.end) for i in scan.islands], [(0, 1), (2, 4)])
        self.assertEqual(scan.islands[1].boundary, NATIVE)
        self.assertEqual(scan.rejections[0].code, "frame_global_extrema")


class FusionSafetyTests(unittest.TestCase):
    """A kernel may only replace a run that already works in its own format.

    Pointwise filters produce different bytes in different pixel formats, so
    fusing a YUV or negotiated run into an RGBA kernel would change the output.
    """

    def test_refuses_an_rgb_filter_inside_a_supported_yuv410_run(self) -> None:
        analysis = analyze_filtergraph(
            "format=yuv410p,negate,lutrgb=r=val*2", auto_islands=True
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "no_profitable_island")
        self.assertIn("removes no frame pass", analysis.diagnostics[0].message)

    def test_refuses_a_yuv_run_containing_an_rgb_only_filter(self) -> None:
        # lutrgb is RGB-only upstream, so FFmpeg converts around it and the run
        # is not contiguous in yuv420p at all. Only negate survives, which
        # removes no frame pass.
        analysis = analyze_filtergraph(
            "format=yuv420p,negate,lutrgb=r=val*2", auto_islands=True
        )
        self.assertFalse(analysis.eligible)
        codes = [diagnostic.code for diagnostic in analysis.diagnostics]
        self.assertIn("format_not_advertised", codes)

    def test_refuses_a_run_whose_format_negotiation_decides(self) -> None:
        analysis = analyze_filtergraph(
            "scale=1280:720,negate,lutrgb=r=val*2", auto_islands=True
        )
        self.assertFalse(analysis.eligible)
        self.assertIn("negotiation", analysis.diagnostics[0].message)

    def test_selection_keeps_only_fusible_profitable_islands(self) -> None:
        scan = scan_chain(chain("negate,scale=2:2,negate,negate"), entry_format="rgba")
        selected = select_islands((scan,))
        # The leading single filter saves nothing; the trailing pair is negotiated.
        self.assertEqual(selected, ())


class AutoIslandAnalysisTests(unittest.TestCase):
    def test_fuses_without_a_trailing_format_boundary(self) -> None:
        source = "format=rgba,negate,lutrgb=r=val*2,negate"
        self.assertFalse(analyze_filtergraph(source).eligible)
        analysis = analyze_filtergraph(source, auto_islands=True)
        self.assertTrue(analysis.eligible)
        self.assertEqual(analysis.region, (1, 4))
        self.assertEqual(analysis.eliminated_passes, 2)

    def test_plans_and_rewrites_several_islands(self) -> None:
        analysis = analyze_filtergraph(
            "format=rgba,negate,lutrgb=r=val*2,crop=64:64,"
            "colorlevels=rimin=0.1,negate,format=yuv420p",
            auto_islands=True,
        )
        self.assertTrue(analysis.eligible)
        self.assertEqual([plan.region for plan in analysis.plans], [(1, 3), (4, 6)])
        self.assertEqual(analysis.eliminated_passes, 2)
        # Distinct regions must not collapse onto one cache key.
        self.assertNotEqual(
            analysis.plans[0].ir.plan_hash, analysis.plans[1].ir.plan_hash
        )
        rewritten = analysis.rewritten_filtergraph or ""
        self.assertEqual(rewritten.count("fused="), 2)
        self.assertIn("crop=64:64", rewritten)
        self.assertTrue(rewritten.startswith("format=rgba,"))
        self.assertTrue(rewritten.endswith(",format=yuv420p"))
        # Several islands have no single IR, so the compatibility view is empty.
        self.assertIsNone(analysis.ir)
        self.assertIsNone(analysis.region)

    def test_explicit_mode_is_unchanged_by_island_discovery(self) -> None:
        source = "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"
        explicit = analyze_filtergraph(source)
        auto = analyze_filtergraph(source, auto_islands=True)
        self.assertTrue(explicit.eligible)
        self.assertEqual(explicit.region, (1, 3))
        self.assertEqual(auto.ir.plan_hash, explicit.ir.plan_hash)


class ScriptParserTests(unittest.TestCase):
    def test_parses_labels_chains_and_whitespace(self) -> None:
        script = parse_filtergraph_script(
            "[0:v]scale=1280:720,format=rgba, negate[mid];[mid][1:v]overlay=x=10[out]"
        )
        self.assertEqual(len(script.chains), 2)
        first = script.chains[0].filters
        self.assertEqual([item.name for item in first], ["scale", "format", "negate"])
        self.assertEqual(first[0].inputs, ("0:v",))
        self.assertEqual(first[2].outputs, ("mid",))
        second = script.chains[1].filters
        self.assertEqual(second[0].inputs, ("mid", "1:v"))

    def test_records_an_option_problem_instead_of_failing_the_graph(self) -> None:
        script = parse_filtergraph_script("negate,lutrgb=r=val:r=negval,hflip")
        names = [item.name for item in script.chains[0].filters]
        self.assertEqual(names, ["negate", "lutrgb", "hflip"])
        self.assertIsNotNone(script.chains[0].filters[1].option_error)

    def test_a_filter_with_unparsed_options_cannot_join_an_island(self) -> None:
        script = parse_filtergraph_script("negate,lutrgb=r=val:r=negval,negate")
        scan = scan_chain(script.chains[0].filters, entry_format="rgba")
        self.assertEqual([(i.start, i.end) for i in scan.islands], [(0, 1), (2, 3)])
        self.assertEqual(scan.rejections[0].code, "unparsed_options")

    def test_strict_parser_still_rejects_graph_syntax(self) -> None:
        from lavfi_cc.parser import FiltergraphSyntaxError

        for source in ("negate;lutrgb", "[in]negate[out]"):
            with self.subTest(source=source):
                with self.assertRaises(FiltergraphSyntaxError):
                    parse_filtergraph(source)


if __name__ == "__main__":
    unittest.main()
