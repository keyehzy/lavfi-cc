from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lavfi_cc.bundle import Bundle, BundleError, build_bundle, resolve_bundle
from lavfi_cc.frontend import analyze_filtergraph


ROOT = Path(__file__).resolve().parents[1]
WEEK5_CANDIDATES = (
    ROOT / ".build" / "ffmpeg-week5-macos" / "bin" / "ffmpeg",
    ROOT / ".build" / "ffmpeg-week5-linux" / "bin" / "ffmpeg",
)
GRAPH = "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"


def week5_ffmpeg_path() -> Path | None:
    override = os.environ.get("LAVFI_CC_WEEK5_FFMPEG") or os.environ.get(
        "LAVFI_CC_WEEK6_FFMPEG"
    )
    candidates = (Path(override),) if override else WEEK5_CANDIDATES
    return next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )


class BuildBundleTests(unittest.TestCase):
    def test_builds_one_kernel_per_distinct_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle([GRAPH, GRAPH], directory)
            self.assertEqual(report.graphs, 2)
            # The same plan twice must not be compiled twice.
            self.assertEqual(len(report.entries), 1)
            self.assertEqual(report.compiled, 1)
            entry = report.entries[0]
            self.assertEqual(entry.graphs, (GRAPH,))
            self.assertEqual(entry.layout, "rgba")
            self.assertTrue((Path(directory) / entry.source_name).is_file())
            self.assertTrue((Path(directory) / (entry.library_name or "")).is_file())

    def test_records_graphs_with_nothing_to_fuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle(["scale=2:2,format=yuv420p"], directory)
            self.assertEqual(report.entries, [])
            self.assertEqual(len(report.ineligible), 1)

    def test_emit_only_writes_source_without_a_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle([GRAPH], directory, compile_kernels=False)
            self.assertEqual(report.compiled, 0)
            entry = report.entries[0]
            self.assertIsNone(entry.library_name)
            source = (Path(directory) / entry.source_name).read_text(encoding="utf-8")
            self.assertIn("lavfi_compiled_kernel", source)
            bundle = Bundle(directory)
            with self.assertRaisesRegex(BundleError, "without a compiled library"):
                bundle.library_path(entry.plan_hash)

    def test_bundles_every_island_of_a_multi_island_graph(self) -> None:
        graph = (
            "format=rgba,negate,lutrgb=r=val*2,crop=8:8,"
            "colorlevels=rimin=0.1,negate,format=yuv420p"
        )
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle([graph], directory, auto_islands=True)
            self.assertEqual(len(report.entries), 2)


class BundleValidationTests(unittest.TestCase):
    def test_opens_and_validates_a_built_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_bundle([GRAPH], directory)
            plan_hash = analyze_filtergraph(GRAPH).ir.plan_hash
            bundle = Bundle(directory)
            self.assertIn(plan_hash, bundle)
            with bundle.open_kernel(plan_hash, "rgba") as kernel:
                self.assertEqual(kernel.plan_hash, plan_hash)

    def test_rejects_a_tampered_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle([GRAPH], directory)
            library = Path(directory) / (report.entries[0].library_name or "")
            library.write_bytes(library.read_bytes() + b"\0")
            with self.assertRaisesRegex(BundleError, "checksum does not match"):
                Bundle(directory).library_path(report.entries[0].plan_hash)

    def test_rejects_a_wrong_layout_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_bundle([GRAPH], directory)
            with self.assertRaisesRegex(BundleError, "is 'rgba', not 'bgra'"):
                Bundle(directory).open_kernel(report.entries[0].plan_hash, "bgra")

    def test_rejects_an_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_bundle([GRAPH], directory)
            index = Path(directory) / "index.json"
            value = json.loads(index.read_text(encoding="utf-8"))
            value["schema"] = 999
            index.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "schema"):
                Bundle(directory)

    def test_missing_index_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BundleError, "index does not exist"):
                Bundle(directory)

    def test_resolve_reads_the_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_bundle([GRAPH], directory)
            self.assertIsNone(resolve_bundle(None))
            previous = os.environ.get("LAVFI_CC_BUNDLE")
            os.environ["LAVFI_CC_BUNDLE"] = directory
            try:
                self.assertIsNotNone(resolve_bundle(None))
            finally:
                if previous is None:
                    del os.environ["LAVFI_CC_BUNDLE"]
                else:
                    os.environ["LAVFI_CC_BUNDLE"] = previous


@unittest.skipUnless(
    week5_ffmpeg_path(),
    "build scripts/build-ffmpeg-week5.sh to run bundle integration tests",
)
class BundleRunTests(unittest.TestCase):
    ARGUMENTS = (
        "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=5",
        "-vf", GRAPH, "-frames:v", "3", "-f", "rawvideo", "-",
    )

    def run_wrapper(
        self, bundle: str, extra: tuple[str, ...], environment: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        env = {**os.environ, **environment}
        return subprocess.run(
            [
                sys.executable, "-m", "lavfi_cc", "run",
                "--ffmpeg", str(week5_ffmpeg_path()),
                "--bundle", bundle, "--require-fusion", *extra, "--",
                *self.ARGUMENTS,
            ],
            cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )

    def test_a_bundled_kernel_runs_without_any_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_bundle([GRAPH], directory)
            baseline = subprocess.run(
                [str(week5_ffmpeg_path()), *self.ARGUMENTS],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr.decode())
            # No compiler is reachable, so a bundle miss could not be repaired.
            fused = self.run_wrapper(
                directory,
                ("--require-bundle",),
                {"PATH": "/var/empty", "LAVFI_CC_CLANG": "/nonexistent/clang"},
            )
            self.assertEqual(fused.returncode, 0, fused.stderr.decode())
            self.assertEqual(fused.stdout, baseline.stdout)

    def test_require_bundle_fails_on_a_miss_instead_of_compiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_bundle(["format=rgba,negate,negate,format=rgba"], directory)
            result = self.run_wrapper(directory, ("--require-bundle",), {})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"bundle has no kernel", result.stderr)


if __name__ == "__main__":
    unittest.main()
