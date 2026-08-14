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
#: lowering, the one-plane-per-loop walk a subsampled layout needs, the
#: two-planes-in-one-loop walk hue's chroma rotation forces, the inline float32
#: expression, which is the only shape that calls libm and converts a float to
#: an integer -- both of which UBSan has something to say about -- and the
#: yuva420p mix of loops at two different resolutions in one kernel, where a
#: plane-3 pointer walked at a chroma plane's row count would run off the end of
#: the frame. The dimensions are odd so a chroma plane's AV_CEIL_RSHIFT rounding
#: is exercised.
#:
#: The last two cases are the 16-bit sample walk, where a row is twice as many
#: bytes as it is samples, and the 10-bit one, whose tables cover 1024 entries
#: while the sample type holds 65536 values. The frames below are random bytes
#: rather than samples in the format's domain, deliberately: that is what makes
#: ASan check the clamp that keeps a stray sample from indexing past the end of
#: a table, and the interpreter clamps identically so the comparison still holds.
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
    (
        "format=gbrap,curves=preset=vintage:interp=pchip,"
        "colorbalance=rs=.3:gm=-.2:bh=.5:bs=-.35,"
        "colorcontrast=rc=.4:gm=.25:by=-.15:rcw=.9:gmw=.7:byw=.3:pl=.35,"
        "format=gbrap",
        257,
        3,
    ),
    (
        "format=yuva420p,lutyuv=y=negval:a=val*0.75+16,"
        "eq=contrast=1.3:saturation=0.8,hue=h=37.5:s=1.4:b=-0.35,"
        "negate=components=y+u+v+a,format=yuva420p",
        257,
        5,
    ),
    (
        "format=rgba64le,negate=components=r+g+b+a,lutrgb=r=negval:g=val*1.08+2,"
        "colorbalance=rs=.3:gm=-.2:bh=.5,"
        "colorcontrast=rc=.4:gm=.25:by=-.15:rcw=.9:gmw=.7:byw=.3:pl=.35,"
        "format=rgba64le",
        129,
        3,
    ),
    (
        "format=yuva420p10le,negate=components=y+u+v+a,"
        "hue=h=37.5:s=1.4:b=-0.35,format=yuva420p10le",
        129,
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
            "-lm",
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
