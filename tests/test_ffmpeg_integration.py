from __future__ import annotations

import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from lavfi_cc.ffmpeg import (
    FFmpegIntegrationError,
    find_filtergraph_argument,
    fused_filter,
    rewrite_ffmpeg_arguments,
    rewrite_filtergraph,
    run_ffmpeg,
)
from lavfi_cc.frontend import analyze_filtergraph
from lavfi_cc.native import CompilationError, compile_kernel, library_suffix


ROOT = Path(__file__).resolve().parents[1]


def week5_ffmpeg_path() -> Path | None:
    configured = os.environ.get("LAVFI_CC_WEEK5_FFMPEG")
    host = {"Darwin": "macos", "Linux": "linux"}.get(platform.system(), "")
    candidates = (
        (Path(configured),) if configured else
        (ROOT / ".build" / f"ffmpeg-week5-{host}" / "bin" / "ffmpeg",)
    )
    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        probe = subprocess.run(
            [str(candidate), "-hide_banner", "-h", "filter=fused"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "kernel_root" in probe.stdout:
            return candidate
    return None


class CommandRewriteTests(unittest.TestCase):
    def test_finds_exactly_one_supported_filter_option(self) -> None:
        short = find_filtergraph_argument(("-i", "in", "-vf", "negate", "out"))
        long = find_filtergraph_argument(("-filter:v", "format=rgba", "out"))
        self.assertEqual((short.option_index, short.value_index), (2, 3))
        self.assertEqual(long.value, "format=rgba")

    def test_rejects_missing_ambiguous_and_attached_forms(self) -> None:
        with self.assertRaisesRegex(FFmpegIntegrationError, "no -vf"):
            find_filtergraph_argument(("-i", "in", "-vf=format=rgba", "out"))
        with self.assertRaisesRegex(FFmpegIntegrationError, "missing"):
            find_filtergraph_argument(("-i", "in", "-vf"))
        with self.assertRaisesRegex(FFmpegIntegrationError, "multiple"):
            find_filtergraph_argument(("-vf", "a", "-filter:v", "b"))

    def test_rewrite_preserves_boundaries_and_opaque_outer_filters(self) -> None:
        graph = "scale=8:8,format=rgba,lutrgb=r=val,format=yuv420p,fps=24"
        analysis = analyze_filtergraph(graph)
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        replacement = fused_filter(analysis, "/private/k.so", "/private")
        rewritten = rewrite_filtergraph(analysis, replacement)
        self.assertTrue(rewritten.startswith("scale=8:8,format=rgba,fused="))
        self.assertTrue(rewritten.endswith(",format=yuv420p,fps=24"))
        self.assertIn(f"plan_hash={analysis.ir.plan_hash}", rewritten)
        self.assertIn("remove_color_side_data=1", rewritten)

        location = find_filtergraph_argument(("-i", "in", "-vf", graph, "out"))
        arguments = rewrite_ffmpeg_arguments(
            ("-i", "in", "-vf", graph, "out"), location, rewritten
        )
        self.assertEqual(arguments[:3], ["-i", "in", "-vf"])
        self.assertEqual(arguments[3], rewritten)
        self.assertEqual(arguments[4], "out")

    def test_filter_value_quotes_apostrophes(self) -> None:
        analysis = analyze_filtergraph("format=rgba,negate,format=rgba")
        rendered = fused_filter(analysis, "/tmp/it's/a\\b.so", "/tmp/it's")
        self.assertIn("'\\''", rendered)
        self.assertIn("a\\\\b.so", rendered)
        self.assertIn("remove_color_side_data=0", rendered)


class FallbackTests(unittest.TestCase):
    @mock.patch("lavfi_cc.ffmpeg.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("lavfi_cc.ffmpeg._execute", return_value=23)
    def test_unsupported_graph_runs_original_by_default(
        self, execute: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-i", "in", "-vf", "format=rgba,scale=2:2,format=rgba", "out")
        self.assertEqual(run_ffmpeg(arguments), 23)
        execute.assert_called_once_with("/fake/ffmpeg", arguments)

    @mock.patch("lavfi_cc.ffmpeg.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("lavfi_cc.ffmpeg._execute")
    def test_require_fusion_rejects_without_running_original(
        self, execute: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-vf", "format=rgba,scale=2:2,format=rgba")
        self.assertEqual(run_ffmpeg(arguments, require_fusion=True), 2)
        execute.assert_not_called()

    @mock.patch("lavfi_cc.ffmpeg.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("lavfi_cc.ffmpeg.compile_kernel", side_effect=CompilationError("boom"))
    @mock.patch("lavfi_cc.ffmpeg._execute", return_value=19)
    def test_compilation_failure_falls_back_to_the_original_command(
        self, execute: mock.Mock, _compile: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-vf", "format=rgba,negate,format=rgba")
        self.assertEqual(run_ffmpeg(arguments), 19)
        execute.assert_called_once_with("/fake/ffmpeg", arguments)

    @mock.patch("lavfi_cc.ffmpeg.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("lavfi_cc.ffmpeg.compile_kernel", side_effect=CompilationError("boom"))
    @mock.patch("lavfi_cc.ffmpeg._execute")
    def test_compilation_failure_is_fatal_in_strict_mode(
        self, execute: mock.Mock, _compile: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-vf", "format=rgba,negate,format=rgba")
        self.assertEqual(run_ffmpeg(arguments, require_fusion=True), 1)
        execute.assert_not_called()

    @mock.patch("lavfi_cc.ffmpeg.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("lavfi_cc.ffmpeg._execute", return_value=29)
    def test_filter_preflight_failure_falls_back_to_original(
        self, execute: mock.Mock, _resolve: mock.Mock
    ) -> None:
        def fake_compile(_ir: object, output: Path, **options: object) -> None:
            Path(output).touch()
            Path(options["source_path"]).touch()

        arguments = ("-vf", "format=rgba,negate,format=rgba")
        with (
            mock.patch("lavfi_cc.ffmpeg.compile_kernel", side_effect=fake_compile),
            mock.patch("lavfi_cc.ffmpeg.NativeKernel"),
            mock.patch(
                "lavfi_cc.ffmpeg._preflight", return_value=(False, "load error")
            ),
        ):
            self.assertEqual(run_ffmpeg(arguments), 29)
        execute.assert_called_once_with("/fake/ffmpeg", arguments)


@unittest.skipUnless(
    week5_ffmpeg_path(),
    "build scripts/build-ffmpeg-week5.sh or set LAVFI_CC_WEEK5_FFMPEG",
)
class FFmpegEndToEndTests(unittest.TestCase):
    @classmethod
    def run_lavfi(
        cls, graph: str, *, wrapper: bool
    ) -> subprocess.CompletedProcess[str]:
        ffmpeg_arguments = [
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=4x4:r=1",
            "-vf",
            graph,
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        command = (
            [
                sys.executable,
                "-m",
                "lavfi_cc",
                "run",
                "--ffmpeg",
                str(week5_ffmpeg_path()),
                "--require-fusion",
                "--",
            ]
            if wrapper
            else [str(week5_ffmpeg_path())]
        )
        return subprocess.run(
            [*command, *ffmpeg_arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @classmethod
    def run_filter_only(cls, filtergraph: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                str(week5_ffmpeg_path()),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=2x2:r=1,format=rgba",
                "-vf",
                filtergraph,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    @classmethod
    def run_raw(
        cls, graph: str, source: bytes, width: int, height: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                str(week5_ffmpeg_path()),
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
                "4",
                "-vf",
                graph,
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
    def run_wrapper(
        cls, graph: str, source: bytes, width: int, height: int
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "lavfi_cc",
                "run",
                "--ffmpeg",
                str(week5_ffmpeg_path()),
                "--require-fusion",
                "--",
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
                "4",
                "-vf",
                graph,
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

    def test_wrapper_is_bit_exact_for_supported_multistage_chains(self) -> None:
        generator = random.Random(0xF05ED)
        cases = (
            (1, 3, "negate=components=r+g+b+a"),
            (17, 5, "lutrgb=r=val*1.125-7:g=negval:b='clip(val,13,241)':a=val"),
            (
                65,
                4,
                "negate,lutrgb=r=val*1.08+2:g=val*.94+4:b=val*.88+12,"
                "colorlevels=rimin=.05:gimax=.9:preserve=none,"
                "colorchannelmixer=rr=.9:rg=.1:gg=.8:gb=.2:bb=1:aa=1:pc=none",
            ),
        )
        for width, height, chain in cases:
            graph = f"format=rgba,{chain},format=rgba"
            source = bytes(
                generator.randrange(256) for _ in range(width * height * 4)
            )
            with self.subTest(width=width, height=height, chain=chain):
                baseline = self.run_raw(graph, source, width, height)
                fused = self.run_wrapper(graph, source, width, height)
                self.assertEqual(
                    baseline.returncode,
                    0,
                    baseline.stderr.decode(errors="replace")[-2000:],
                )
                self.assertEqual(
                    fused.returncode,
                    0,
                    fused.stderr.decode(errors="replace")[-2000:],
                )
                self.assertEqual(len(fused.stdout), width * height * 4)
                self.assertEqual(fused.stdout, baseline.stdout)

    def test_filter_rejects_world_writable_and_hash_mismatched_kernels(self) -> None:
        analysis = analyze_filtergraph("format=rgba,negate,format=rgba")
        self.assertTrue(analysis.eligible, analysis.diagnostics)
        with tempfile.TemporaryDirectory(prefix="lavfi-cc-filter-security-") as temp:
            directory = Path(temp)
            library = directory / (analysis.ir.plan_hash + library_suffix())
            compile_kernel(analysis.ir, library)
            replacement = fused_filter(analysis, library, directory)

            os.chmod(library, 0o666)
            writable = self.run_filter_only(replacement)
            self.assertNotEqual(writable.returncode, 0)
            self.assertIn(b"world-writable", writable.stderr)

            os.chmod(library, 0o600)
            wrong_hash = "0" * 64
            if wrong_hash == analysis.ir.plan_hash:
                wrong_hash = "1" * 64
            mismatched = self.run_filter_only(
                replacement.replace(
                    f"plan_hash={analysis.ir.plan_hash}",
                    f"plan_hash={wrong_hash}",
                )
            )
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn(b"plan hash does not match", mismatched.stderr)

    def test_fused_filter_preserves_timestamps_color_fields_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lavfi-cc-frame-props-") as temp:
            directory = Path(temp)
            prefix = (
                "setpts=PTS+7/TB,"
                "setparams=range=full:color_primaries=bt709:color_trc=bt709:"
                "colorspace=bt709,"
                "metadata=mode=add:key=lavfi_cc:value=preserved,format=rgba,"
                "negate,format=rgba,showinfo,metadata=mode=print:file="
            )
            baseline_metadata = directory / "baseline.txt"
            fused_metadata = directory / "fused.txt"
            baseline = self.run_lavfi(prefix + str(baseline_metadata), wrapper=False)
            fused = self.run_lavfi(prefix + str(fused_metadata), wrapper=True)
            self.assertEqual(baseline.returncode, 0, baseline.stderr[-2000:])
            self.assertEqual(fused.returncode, 0, fused.stderr[-2000:])
            self.assertEqual(
                fused_metadata.read_text(), baseline_metadata.read_text()
            )
            self.assertIn("lavfi_cc=preserved", fused_metadata.read_text())

            def showinfo_properties(stderr: str) -> list[str]:
                return [
                    line.split("] ", 1)[1]
                    for line in stderr.splitlines()
                    if "Parsed_showinfo" in line
                    and any(
                        marker in line
                        for marker in (" n:", "color_range:", "alpha_mode:")
                    )
                ]

            properties = showinfo_properties(fused.stderr)
            self.assertEqual(properties, showinfo_properties(baseline.stderr))
            self.assertTrue(any("pts:      7" in item for item in properties))
            self.assertTrue(any("color_primaries:bt709" in item for item in properties))


if __name__ == "__main__":
    unittest.main()
