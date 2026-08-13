from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from lavfi_cc.scanner import load_graphs, scan_graph, summarize


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus" / "filtergraphs.txt"


class ScanGraphTests(unittest.TestCase):
    def test_reports_stages_and_passes_for_a_fusible_island(self) -> None:
        scan = scan_graph("format=rgba,negate,lutrgb=r=val*2,negate,format=yuv420p")
        self.assertTrue(scan.parsed)
        self.assertEqual(len(scan.islands), 1)
        report = scan.islands[0]
        self.assertTrue(report.fusible)
        self.assertEqual(report.stages, 3)
        # Three composable LUT stages collapse to one.
        self.assertEqual(report.optimized_stages, 1)
        self.assertEqual(scan.eliminated_passes, 2)
        self.assertEqual(scan.blocked_passes, 0)

    def test_attributes_a_blocked_island_to_its_working_format(self) -> None:
        scan = scan_graph("format=yuv410p,negate,lutrgb=r=val*2,negate")
        self.assertEqual(scan.eliminated_passes, 0)
        self.assertEqual(scan.blocked_passes, 2)
        self.assertEqual(scan.islands[0].island.working_format, "yuv410p")

    def test_a_supported_yuv_island_is_fusible(self) -> None:
        scan = scan_graph("format=yuv420p,negate,negate,negate")
        self.assertEqual(scan.eliminated_passes, 2)
        self.assertEqual(scan.blocked_passes, 0)
        self.assertTrue(scan.islands[0].fusible)
        # Three negations compose into one, which is itself not the identity.
        self.assertEqual(scan.islands[0].optimized_stages, 1)

    def test_scans_several_chains_with_link_labels(self) -> None:
        scan = scan_graph(
            "[0:v]format=rgba,negate,negate[a];"
            "[1:v]format=yuv410p,negate,negate[b];"
            "[a][b]overlay[out]"
        )
        self.assertEqual(len(scan.chains), 3)
        self.assertEqual([r.island.chain_index for r in scan.islands], [0, 1])
        self.assertEqual(scan.eliminated_passes, 1)
        self.assertEqual(scan.blocked_passes, 1)

    def test_an_unreadable_option_does_not_fail_the_scan(self) -> None:
        scan = scan_graph("negate,lutrgb=r=val:r=negval,negate")
        self.assertTrue(scan.parsed)
        codes = [
            rejection.code for chain in scan.chains for rejection in chain.rejections
        ]
        self.assertIn("unparsed_options", codes)

    def test_a_bad_option_list_stays_local_to_its_filter(self) -> None:
        scan = scan_graph("negate,negate=,negate")
        self.assertTrue(scan.parsed)
        self.assertEqual(scan.chains[0].filter_count, 3)
        self.assertEqual([(i.island.start, i.island.end) for i in scan.islands],
                         [(0, 1), (2, 3)])

    def test_reports_a_structural_syntax_error_without_raising(self) -> None:
        scan = scan_graph("negate,lutrgb=r='unterminated")
        self.assertFalse(scan.parsed)
        self.assertIsNotNone(scan.error)
        self.assertIn("quote", scan.error or "")

    def test_entry_format_lets_a_caller_prove_the_source_format(self) -> None:
        graph = "negate,lutrgb=r=val*2,negate"
        self.assertEqual(scan_graph(graph).eliminated_passes, 0)
        self.assertEqual(
            scan_graph(graph, entry_format="rgba").eliminated_passes, 2
        )


class SummaryTests(unittest.TestCase):
    def test_ranks_blockers_by_the_passes_they_withhold(self) -> None:
        summary = summarize(
            [
                scan_graph("format=yuv410p,negate,negate,negate,negate"),
                scan_graph("format=yuv411p,negate,negate"),
                scan_graph("format=rgba,negate,negate"),
            ]
        )
        self.assertEqual(summary.graphs, 3)
        self.assertEqual(summary.eliminated_passes, 1)
        self.assertEqual(summary.blocked_passes, 4)
        # yuv410p withholds three passes against yuv411p's one, so it ranks first.
        self.assertEqual(
            summary.blocked_passes_by_format.most_common(1), [("yuv410p", 3)]
        )

    def test_counts_unsupported_filters_and_rejected_options_apart(self) -> None:
        summary = summarize(
            [
                scan_graph("format=rgba,negate,gblur=sigma=3,negate"),
                scan_graph("format=rgba,colorlevels=rimin=-0.1"),
            ]
        )
        self.assertEqual(summary.unsupported_filters["gblur"], 1)
        self.assertEqual(
            summary.rejected_options["colorlevels:frame_global_extrema"], 1
        )


class CorpusTests(unittest.TestCase):
    def test_load_graphs_skips_blanks_and_comments(self) -> None:
        graphs = load_graphs(["# note", "", "  negate  ", "lutrgb=r=val"])
        self.assertEqual(graphs, ["negate", "lutrgb=r=val"])

    def test_every_corpus_graph_parses(self) -> None:
        graphs = load_graphs(CORPUS.read_text(encoding="utf-8").splitlines())
        self.assertGreater(len(graphs), 20)
        scans = [scan_graph(graph) for graph in graphs]
        unparsed = [(scan.source, scan.error) for scan in scans if not scan.parsed]
        self.assertEqual(unparsed, [])

    def test_corpus_finds_both_fusible_and_blocked_islands(self) -> None:
        graphs = load_graphs(CORPUS.read_text(encoding="utf-8").splitlines())
        summary = summarize(scan_graph(graph) for graph in graphs)
        self.assertGreater(summary.fusible_islands, 0)
        self.assertGreater(summary.eliminated_passes, 0)
        self.assertGreater(summary.blocked_passes, 0)
        self.assertIn("scale", summary.unsupported_filters)


class ScanCommandTests(unittest.TestCase):
    @staticmethod
    def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "lavfi_cc", "scan", *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_scan_reports_a_single_graph_in_detail(self) -> None:
        result = self.run_cli("--vf", "format=rgba,negate,negate,format=yuv420p")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fusible", result.stdout)
        self.assertIn("frame passes eliminated:   1", result.stdout)

    def test_scan_never_compiles_a_kernel(self) -> None:
        # A cache directory that cannot be created would break any compile path.
        result = self.run_cli(
            "--file", str(CORPUS), "--json", "--entry-format", "rgba"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"summary"', result.stdout)

    def test_scan_requires_a_source(self) -> None:
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
