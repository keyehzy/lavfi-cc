from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "lavfi_cc", *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_explain_success(self) -> None:
        result = self.run_cli(
            "explain", "--vf", "format=rgba,negate,format=yuv420p"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Eligibility: eligible", result.stdout)
        self.assertIn("plan_hash:", result.stdout)
        self.assertIn("Planned rewrite:", result.stdout)

    def test_explain_failure_is_nonzero_and_precise(self) -> None:
        result = self.run_cli(
            "explain", "--vf", "format=rgba,scale=2:2,format=yuv420p"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported_filter", result.stdout)
        self.assertIn("filter[1]", result.stdout)

    def test_json_exposes_debug_and_canonical_ir(self) -> None:
        result = self.run_cli(
            "explain", "--json", "--vf", "format=rgba,negate,format=yuv420p"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertTrue(value["eligible"])
        self.assertIn("source", value["ir"]["operations"][1])
        self.assertNotIn("source", value["canonical_ir"]["operations"][1])
        self.assertIn("optimization", value)

    def test_interpret_streams_multiple_raw_frames(self) -> None:
        source = bytes((0, 1, 2, 3, 4, 5, 6, 7))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lavfi_cc",
                "interpret",
                "--vf",
                "format=rgba,negate=components=r+g+b+a,format=rgba",
                "--width",
                "1",
                "--height",
                "1",
            ],
            cwd=ROOT,
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, bytes(255 - value for value in source))

    def test_interpret_rejects_a_partial_frame(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lavfi_cc",
                "interpret",
                "--vf",
                "format=rgba,negate,format=rgba",
                "--width",
                "2",
                "--height",
                "1",
            ],
            cwd=ROOT,
            input=b"123",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"partial frame 0", result.stderr)

    def test_interpret_reads_and_writes_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.rgba"
            output = Path(directory) / "output.rgba"
            source.write_bytes(bytes((1, 2, 3, 4)))
            result = self.run_cli(
                "interpret",
                "--vf",
                "format=rgba,negate=components=a,format=rgba",
                "--width",
                "1",
                "--height",
                "1",
                "--input",
                str(source),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), bytes((1, 2, 3, 251)))

    @unittest.skipUnless(shutil.which("clang"), "Week 4 CLI tests require Clang")
    def test_native_streams_multiple_raw_frames(self) -> None:
        source = bytes((0, 1, 2, 3, 4, 5, 6, 7))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lavfi_cc",
                "native",
                "--vf",
                "format=rgba,negate=components=r+g+b+a,format=rgba",
                "--width",
                "1",
                "--height",
                "1",
            ],
            cwd=ROOT,
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, bytes(255 - value for value in source))

    @unittest.skipUnless(shutil.which("clang"), "Week 4 CLI tests require Clang")
    def test_compile_writes_a_checked_library_and_readable_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernel.dylib"
            source = Path(directory) / "kernel.c"
            result = self.run_cli(
                "compile",
                "--vf",
                "format=rgba,negate,lutrgb=r=val,format=rgba",
                "--output",
                str(output),
                "--emit-c",
                str(source),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(source.is_file())
            self.assertIn("Native library:", result.stdout)
            self.assertIn("lavfi_compiled_kernel", source.read_text())

    def test_run_requires_the_double_dash_separator(self) -> None:
        result = self.run_cli("run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be separated with --", result.stderr)


if __name__ == "__main__":
    unittest.main()
