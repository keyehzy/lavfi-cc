from __future__ import annotations

import csv
import os
from pathlib import Path
import random
import re
import subprocess
import unittest

from lavfi_cc.frontend import analyze_filtergraph
from lavfi_cc.interpreter import interpret_rgba8
from lavfi_cc.native import NativeKernel


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

    @classmethod
    def run_raw_frame(
        cls, filtergraph: str, source: bytes, width: int, height: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                str(ffmpeg_path()),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "1",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-filter_threads",
                "1",
                "-vf",
                filtergraph,
                "-pix_fmt",
                "rgba",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            cwd=ROOT,
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @classmethod
    def run_source_frame(
        cls, source_filtergraph: str, width: int, height: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                str(ffmpeg_path()),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                source_filtergraph,
                "-frames:v",
                "1",
                "-filter_threads",
                "1",
                "-pix_fmt",
                "rgba",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def assert_interpreter_matches(
        self, chain: str, source: bytes, width: int, height: int
    ) -> None:
        graph = f"format=rgba,{chain},format=rgba"
        analysis = analyze_filtergraph(graph)
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        assert analysis.ir is not None
        expected_process = self.run_raw_frame(graph, source, width, height)
        self.assertEqual(
            expected_process.returncode,
            0,
            expected_process.stderr.decode(errors="replace")[-2000:],
        )
        expected = expected_process.stdout
        observed = interpret_rgba8(analysis.ir, source, width, height)
        with NativeKernel.compile(analysis.ir) as kernel:
            native = kernel.process_rgba8(source, width, height)
        self.assertEqual(len(expected), width * height * 4)
        if observed != expected:
            difference = next(
                index
                for index, (actual, oracle) in enumerate(zip(observed, expected, strict=True))
                if actual != oracle
            )
            pixel, channel = divmod(difference, 4)
            x, y = pixel % width, pixel // width
            self.fail(
                f"byte difference for {chain!r} at pixel ({x}, {y}) channel "
                f"{'rgba'[channel]}: interpreter={observed[difference]}, "
                f"ffmpeg={expected[difference]}"
            )
        if native != expected:
            difference = next(
                index
                for index, (actual, oracle) in enumerate(zip(native, expected, strict=True))
                if actual != oracle
            )
            pixel, channel = divmod(difference, 4)
            x, y = pixel % width, pixel // width
            self.fail(
                f"byte difference for {chain!r} at pixel ({x}, {y}) channel "
                f"{'rgba'[channel]}: native={native[difference]}, "
                f"ffmpeg={expected[difference]}"
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

    def test_interpreter_matches_every_versioned_corpus_chain(self) -> None:
        def source_for(pattern: str, width: int, height: int) -> str:
            if pattern in {"black", "white"}:
                return f"color=c={pattern}:size={width}x{height}:rate=1"
            if pattern in {"testsrc", "testsrc2"}:
                return f"{pattern}=size={width}x{height}:rate=1"
            if pattern == "rgba_ramp":
                return (
                    f"nullsrc=size={width}x{height}:rate=1,format=rgba,"
                    "geq=r='X/W*255':g='Y/H*255':b='(X+Y)/(W+H)*255':"
                    "a='X/W*255',format=rgba"
                )
            if pattern == "checker":
                return (
                    f"nullsrc=size={width}x{height}:rate=1,format=rgba,"
                    "geq=r='mod(X*37+Y*17,256)':g='mod(X*11+Y*43,256)':"
                    "b='mod(X*29+Y*7,256)':a='mod(X*13+Y*19,256)',format=rgba"
                )
            raise AssertionError(f"unknown corpus pattern {pattern!r}")

        with (ROOT / "tests/corpus/cases.tsv").open(newline="") as handle:
            rows = csv.reader(handle, delimiter="\t")
            next(rows)
            for row in rows:
                identifier, width_text, height_text, pattern, chain = row
                width, height = int(width_text), int(height_text)
                source_process = self.run_source_frame(
                    source_for(pattern, width, height), width, height
                )
                with self.subTest(identifier=identifier):
                    self.assertEqual(
                        source_process.returncode,
                        0,
                        source_process.stderr.decode(errors="replace")[-2000:],
                    )
                    self.assertEqual(len(source_process.stdout), width * height * 4)
                    self.assert_interpreter_matches(
                        chain, source_process.stdout, width, height
                    )

    def test_interpreter_matches_all_channel_values_and_parameter_edges(self) -> None:
        source = bytearray()
        for value in range(256):
            source.extend((value, (value * 73) & 0xFF, 255 - value, (value * 151) & 0xFF))
        chains = (
            "negate=components=r+g+b+a",
            "lutrgb=r=val*1.5-2.75:g=negval:b='clip(val,17,239)':a=val/2",
            "colorlevels=rimin=1:rimax=0:romin=0.9:romax=0.1:"
            "gimin=0:gimax=1:gomin=1:gomax=0:preserve=none",
            "colorchannelmixer=rr=-2:rg=2:rb=0.5:ra=-0.5:"
            "gr=1.5:gg=-1.5:gb=0.25:ga=0.75:"
            "br=-0.1:bg=0.1:bb=1:ba=0:ar=2:ag=-2:ab=1:aa=0.5:pc=none",
        )
        for chain in chains:
            with self.subTest(chain=chain):
                self.assert_interpreter_matches(chain, bytes(source), 256, 1)

    def test_interpreter_matches_deterministically_generated_chains(self) -> None:
        generator = random.Random(0x1A7F1)
        widths = (1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 1919, 1920, 3840)
        lut_expressions = (
            "val",
            "negval",
            "val*0.5+3",
            "val*1.125-7",
            "maxval-val/3",
        )
        coefficients = (-2.0, -0.75, -0.5, 0.0, 0.25, 0.5, 0.9, 1.0, 1.5, 2.0)

        def generated_filter(kind: int) -> str:
            if kind == 0:
                component_sets = ("r", "a", "r+g+b", "r+g+b+a", "b+r")
                return "negate=components=" + generator.choice(component_sets)
            if kind == 1:
                return "lutrgb=" + ":".join(
                    f"{channel}={generator.choice(lut_expressions)}" for channel in "rgba"
                )
            if kind == 2:
                imin, imax = generator.sample((0.0, 0.1, 0.25, 0.75, 0.9, 1.0), 2)
                omin, omax = generator.sample((0.0, 0.2, 0.8, 1.0), 2)
                return (
                    f"colorlevels=rimin={imin}:rimax={imax}:romin={omin}:romax={omax}:"
                    "preserve=none"
                )
            names = ("rr", "rg", "rb", "gr", "gg", "gb", "br", "bg", "bb", "aa")
            options = ":".join(
                f"{name}={generator.choice(coefficients)}" for name in names
            )
            return f"colorchannelmixer={options}:pc=none"

        for case_index, width in enumerate(widths):
            stage_count = 1 + case_index % 6
            chain = ",".join(
                generated_filter(generator.randrange(4)) for _ in range(stage_count)
            )
            height = 1 if case_index % 2 == 0 else 2
            source = bytes(generator.randrange(256) for _ in range(width * height * 4))
            with self.subTest(case_index=case_index, width=width, chain=chain):
                self.assert_interpreter_matches(chain, source, width, height)


if __name__ == "__main__":
    unittest.main()
