#!/usr/bin/env python3
"""Build and execute a representative kernel under ASan and UBSan."""

from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lavfi_cc.codegen import generate_c
from lavfi_cc.frontend import require_ir
from lavfi_cc.interpreter import interpret_rgba8
from lavfi_cc.layouts import get_layout


#: One case per code-generation shape: the packed whole-pixel walk with every
#: lowering, the one-plane-per-loop walk a subsampled layout needs, and the
#: two-planes-in-one-loop walk hue's chroma rotation forces. The dimensions are
#: odd so a chroma plane's AV_CEIL_RSHIFT rounding is exercised.
CASES = (
    (
        "format=rgba,negate=components=r+g+b+a,"
        "lutrgb=r=val*1.08+2:g=negval:b='clip(val,13,241)':a=val,"
        "colorlevels=rimin=.05:gimax=.9:preserve=none,"
        "colorchannelmixer=rr=.9:rg=.1:gg=.8:gb=.2:bb=1:aa=1:pc=none,"
        "format=rgba",
        257,
        3,
    ),
    (
        "format=yuv420p,negate,negate=components=y,"
        "negate=components=u+v,format=yuv420p",
        257,
        5,
    ),
    (
        "format=yuv420p,lutyuv=y=negval:u=val*0.9+12,"
        "eq=contrast=1.3:saturation=0.8:gamma=1.7,"
        "hue=h=37.5:s=1.4:b=-0.35,format=yuv420p",
        257,
        5,
    ),
)


def sanitize(clang: str, graph: str, width: int, height: int) -> None:
    ir = require_ir(graph)
    layout = get_layout(ir.layout)
    generator = random.Random(0xA5A6)
    source = bytes(
        generator.randrange(256) for _ in range(layout.frame_size(width, height))
    )
    expected = interpret_rgba8(ir, source, width, height)

    with tempfile.TemporaryDirectory(prefix="lavfi-cc-week6-sanitize-") as directory:
        work = Path(directory)
        generated = work / "kernel.c"
        source_path = work / "source.raw"
        expected_path = work / "expected.raw"
        executable = work / "kernel-sanitized"
        generated.write_text(generate_c(ir).source, encoding="utf-8")
        source_path.write_bytes(source)
        expected_path.write_bytes(expected)
        command = [
            clang,
            "-std=c11",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fsanitize=address,undefined",
            "-I",
            str(ROOT / "runtime"),
            str(generated),
            str(ROOT / "tests" / "native_sanitizer_harness.c"),
            "-o",
            str(executable),
        ]
        subprocess.run(command, check=True)
        environment = os.environ.copy()
        environment["ASAN_OPTIONS"] = "halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        subprocess.run(
            [
                str(executable),
                str(source_path),
                str(expected_path),
                str(layout.abi_id),
                str(width),
                str(height),
                # The harness is told each plane's geometry so it needs no copy
                # of the layout table.
                *(
                    f"{layout.plane_height(plane, height)}:"
                    f"{layout.plane_row_bytes(plane, width)}"
                    for plane in range(layout.plane_count)
                ),
            ],
            env=environment,
            check=True,
        )
    print(f"  {ir.layout} {width}x{height}: clean and interpreter-exact")


def main() -> int:
    clang = os.environ.get("LAVFI_CC_CLANG") or shutil.which("clang")
    if not clang:
        raise SystemExit("Week 6 sanitizer gate requires Clang")
    print("Week 6 sanitizer gate: ASan/UBSan")
    for graph, width, height in CASES:
        sanitize(clang, graph, width, height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
