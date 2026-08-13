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


def main() -> int:
    clang = os.environ.get("LAVFI_CC_CLANG") or shutil.which("clang")
    if not clang:
        raise SystemExit("Week 6 sanitizer gate requires Clang")
    graph = (
        "format=rgba,negate=components=r+g+b+a,"
        "lutrgb=r=val*1.08+2:g=negval:b='clip(val,13,241)':a=val,"
        "colorlevels=rimin=.05:gimax=.9:preserve=none,"
        "colorchannelmixer=rr=.9:rg=.1:gg=.8:gb=.2:bb=1:aa=1:pc=none,"
        "format=rgba"
    )
    ir = require_ir(graph)
    width = 257
    height = 3
    generator = random.Random(0xA5A6)
    source = bytes(generator.randrange(256) for _ in range(width * height * 4))
    expected = interpret_rgba8(ir, source, width, height)

    with tempfile.TemporaryDirectory(prefix="lavfi-cc-week6-sanitize-") as directory:
        work = Path(directory)
        generated = work / "kernel.c"
        source_path = work / "source.rgba"
        expected_path = work / "expected.rgba"
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
            [str(executable), str(source_path), str(expected_path)],
            env=environment,
            check=True,
        )
    print("Week 6 sanitizer gate: ASan/UBSan clean and interpreter-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
