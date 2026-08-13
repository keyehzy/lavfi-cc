from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import unittest

from lavfi_cc.frontend import analyze_filtergraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FFMPEG_CANDIDATES = (
    ROOT / ".build" / "ffmpeg-macos" / "bin" / "ffmpeg",
    ROOT / ".build" / "ffmpeg-linux" / "bin" / "ffmpeg",
)


def ffmpeg_path() -> Path | None:
    configured = os.environ.get("LAVFI_CC_FFMPEG")
    candidates = (Path(configured),) if configured else DEFAULT_FFMPEG_CANDIDATES
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file() and os.access(candidate, os.X_OK)
        ),
        None,
    )


@unittest.skipUnless(ffmpeg_path(), "set LAVFI_CC_FFMPEG to run parser differential tests")
class FFmpegDifferentialTests(unittest.TestCase):
    @classmethod
    def run_ffmpeg(cls, filtergraph: str, loglevel: str = "error") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(ffmpeg_path()),
                "-hide_banner",
                "-loglevel",
                loglevel,
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=16x16:r=1",
                "-vf",
                filtergraph,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_every_versioned_accepted_chain_is_accepted_by_pinned_ffmpeg(self) -> None:
        chains: list[tuple[str, str]] = []
        for relative in ("benchmarks/chains.tsv", "tests/corpus/cases.tsv"):
            with (ROOT / relative).open(newline="") as handle:
                rows = csv.reader(handle, delimiter="\t")
                next(rows)
                for row in rows:
                    identifier = row[0]
                    region = row[-1]
                    if region == "null":
                        continue
                    chains.append((identifier, region))

        for identifier, region in chains:
            with self.subTest(identifier=identifier):
                graph = f"format=rgba,{region},format=rgba"
                analysis = analyze_filtergraph(graph)
                self.assertTrue(analysis.eligible, analysis.diagnostics)
                repeated = analyze_filtergraph(graph)
                self.assertEqual(analysis.ir.serialize(), repeated.ir.serialize())
                result = self.run_ffmpeg(graph)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_lut_table_matches_ffmpeg_trace_for_quoted_commas(self) -> None:
        graph = "format=rgba,lutrgb=r='clip(val,0,240)',format=rgba"
        analysis = analyze_filtergraph(graph)
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        result = self.run_ffmpeg(graph, "trace")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        observed: dict[int, int] = {}
        pattern = re.compile(r"Parsed_lutrgb_1.*val\[0\]\[(\d+)\] = (\d+)")
        for line in result.stderr.splitlines():
            match = pattern.search(line)
            if match and int(match.group(1)) < 256:
                # FFmpeg allocates/traces the maximum-depth LUT even when the
                # negotiated frame format is RGBA8; only entries 0..255 are
                # addressable by this region.
                observed[int(match.group(1))] = int(match.group(2))
        self.assertEqual(len(observed), 256)
        table = next(
            operation.parameters["tables"][0]
            for operation in analysis.ir.operations
            if operation.kind == "lut8"
        )
        self.assertEqual(tuple(observed[index] for index in range(256)), table)


if __name__ == "__main__":
    unittest.main()
