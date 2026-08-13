#!/usr/bin/env python3
"""Small reproducible Week 4 native-versus-interpreter exit-gate benchmark."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import statistics
import sys
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lavfi_cc.frontend import require_ir  # noqa: E402
from lavfi_cc.interpreter import interpret_into  # noqa: E402
from lavfi_cc.native import NativeKernel  # noqa: E402


CHAIN = (
    "format=rgba,"
    "negate,"
    "lutrgb=r=val*1.08+2:g=val*.96:b=negval,"
    "colorlevels=rimin=.05:gimax=.9:preserve=none,"
    "colorchannelmixer=rr=.9:rg=.1:gg=.9:gb=.1:bb=.9:br=.1:pc=none,"
    "format=rgba"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def _median_seconds(operation: Callable[[], None], samples: int) -> float:
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        elapsed.append(time.perf_counter() - started)
    return statistics.median(elapsed)


def main() -> int:
    arguments = _arguments()
    if arguments.width <= 0 or arguments.height <= 0 or arguments.samples <= 0:
        raise SystemExit("width, height, and samples must be positive")
    ir = require_ir(CHAIN)
    size = arguments.width * arguments.height * 4
    source = bytearray((index * 73 + 19) & 0xFF for index in range(size))
    interpreted = bytearray(size)
    uncomposed_native = bytearray(size)
    optimized_native = bytearray(size)

    with ExitStack() as stack:
        compile_started = time.perf_counter()
        uncomposed_kernel = stack.enter_context(
            NativeKernel.compile(ir, lut_composition=False)
        )
        uncomposed_compile_seconds = time.perf_counter() - compile_started
        compile_started = time.perf_counter()
        optimized_kernel = stack.enter_context(NativeKernel.compile(ir))
        optimized_compile_seconds = time.perf_counter() - compile_started

        def run_interpreter() -> None:
            interpret_into(
                ir, source, interpreted, arguments.width, arguments.height
            )

        def run_uncomposed_native() -> None:
            uncomposed_kernel.process_into(
                source, uncomposed_native, arguments.width, arguments.height
            )

        def run_optimized_native() -> None:
            optimized_kernel.process_into(
                source, optimized_native, arguments.width, arguments.height
            )

        run_interpreter()
        run_uncomposed_native()
        run_optimized_native()
        if uncomposed_native != interpreted or optimized_native != interpreted:
            raise SystemExit("native output differs from the reference interpreter")
        interpreter_seconds = _median_seconds(run_interpreter, arguments.samples)
        uncomposed_native_seconds = _median_seconds(
            run_uncomposed_native, arguments.samples
        )
        native_seconds = _median_seconds(run_optimized_native, arguments.samples)

    speedup = interpreter_seconds / native_seconds
    optimization_speedup = uncomposed_native_seconds / native_seconds
    pixels = arguments.width * arguments.height
    print(f"plan_hash\t{ir.plan_hash}")
    print(f"dimensions\t{arguments.width}x{arguments.height}")
    print(f"samples\t{arguments.samples}")
    print(
        "uncomposed_cold_compile_and_load_ms\t"
        f"{uncomposed_compile_seconds * 1000:.3f}"
    )
    print(
        "optimized_cold_compile_and_load_ms\t"
        f"{optimized_compile_seconds * 1000:.3f}"
    )
    print(f"interpreter_median_ms\t{interpreter_seconds * 1000:.3f}")
    print(f"uncomposed_native_median_ms\t{uncomposed_native_seconds * 1000:.3f}")
    print(f"native_median_ms\t{native_seconds * 1000:.3f}")
    print(f"interpreter_mpix_per_s\t{pixels / interpreter_seconds / 1e6:.3f}")
    print(f"native_mpix_per_s\t{pixels / native_seconds / 1e6:.3f}")
    print(f"speedup\t{speedup:.2f}x")
    print(f"levels_lut_mixer_fusion_speedup\t{optimization_speedup:.2f}x")
    if native_seconds >= interpreter_seconds:
        print("Week 4 performance gate failed: native was not faster", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
