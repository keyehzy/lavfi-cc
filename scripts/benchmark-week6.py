#!/usr/bin/env python3
"""Measure the incremental warm-cache lookup and validation cost."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lavfi_cc.cache import KernelCache
from lavfi_cc.frontend import require_ir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--max-ms", type=float, default=20.0)
    arguments = parser.parse_args()
    if arguments.samples < 5 or arguments.max_ms <= 0:
        parser.error("--samples must be at least 5 and --max-ms must be positive")

    ir = require_ir(
        "format=rgba,negate,lutrgb=r=val*1.08+2,"
        "colorlevels=gimax=.9:preserve=none,"
        "colorchannelmixer=rr=.9:rg=.1:gg=1:bb=1:aa=1:pc=none,format=rgba"
    )
    with tempfile.TemporaryDirectory(prefix="lavfi-cc-week6-benchmark-") as directory:
        cache = KernelCache(Path(directory) / "cache")
        cold = cache.ensure(ir)
        samples = []
        for _ in range(arguments.samples):
            start = time.perf_counter_ns()
            warm = cache.ensure(ir)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            if warm.status != "hit" or warm.key != cold.key:
                raise RuntimeError("warm cache lookup did not return the compiled entry")
    median = statistics.median(samples)
    print(
        f"Week 6 warm-cache overhead: median={median:.3f} ms "
        f"min={min(samples):.3f} ms max={max(samples):.3f} ms "
        f"samples={len(samples)} target<{arguments.max_ms:.3f} ms"
    )
    return 0 if median < arguments.max_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
