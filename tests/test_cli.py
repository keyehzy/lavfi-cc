from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
