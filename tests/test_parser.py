from __future__ import annotations

import unittest

from lavfi_cc.parser import FiltergraphSyntaxError, parse_filtergraph


class ParserTests(unittest.TestCase):
    def test_parses_linear_graph_and_tracks_offsets(self) -> None:
        source = "format=rgba,negate=components=r+g+b,format=yuv420p"
        graph = parse_filtergraph(source)
        self.assertEqual([item.name for item in graph.filters], ["format", "negate", "format"])
        self.assertEqual(graph.filters[1].span.start, source.index("negate"))
        self.assertEqual(graph.filters[1].named_options()["components"].value, "r+g+b")

    def test_single_quotes_protect_expression_separators(self) -> None:
        graph = parse_filtergraph("lutrgb=r='clip(val, 0, 240)':g='min(val, 200)'")
        options = graph.filters[0].named_options()
        self.assertEqual(options["r"].value, "clip(val, 0, 240)")
        self.assertEqual(options["g"].value, "min(val, 200)")

    def test_backslash_protects_expression_separators(self) -> None:
        graph = parse_filtergraph(r"lutrgb=r=clip(val\,0\,240)")
        self.assertEqual(
            graph.filters[0].named_options()["r"].value, "clip(val,0,240)"
        )

    def test_rejects_non_linear_graph_features(self) -> None:
        for source in ("negate;lutrgb", "[in]negate[out]", "negate,,lutrgb"):
            with self.subTest(source=source):
                with self.assertRaises(FiltergraphSyntaxError):
                    parse_filtergraph(source)

    def test_rejects_ambiguous_duplicate_options(self) -> None:
        with self.assertRaisesRegex(FiltergraphSyntaxError, "duplicate option"):
            parse_filtergraph("lutrgb=r=val:r=negval")

    def test_rejects_double_quotes_with_actionable_message(self) -> None:
        with self.assertRaisesRegex(FiltergraphSyntaxError, "use single quotes"):
            parse_filtergraph('lutrgb=r="min(val,200)"')

    def test_rejects_whitespace_that_ffmpeg_may_reinterpret(self) -> None:
        with self.assertRaisesRegex(FiltergraphSyntaxError, "whitespace"):
            parse_filtergraph("negate, lutrgb")


if __name__ == "__main__":
    unittest.main()
