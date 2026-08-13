#!/usr/bin/env python3
"""Benchmark the ordinary and fused pipelines on a real video input."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHAIN = (
    "format=rgba,"
    "negate,"
    "lutrgb=r=val*1.08+2:g=val*0.94+4:b=val*0.88+12:a=val,"
    "colorlevels=rimin=0.04:gimin=0.02:bimin=0.06:"
    "rimax=0.96:gimax=0.98:bimax=0.94:preserve=none,"
    "colorchannelmixer=rr=0.90:rg=0.08:rb=0.02:"
    "gr=0.03:gg=0.94:gb=0.03:br=0.04:bg=0.06:bb=0.90:pc=none,"
    "format=yuv420p"
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _arguments() -> argparse.Namespace:
    host = {"Darwin": "macos", "Linux": "linux"}.get(platform.system())
    default_ffmpeg = (
        ROOT / ".build" / f"ffmpeg-week5-{host}" / "bin" / "ffmpeg"
        if host
        else Path("ffmpeg")
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "video.mp4")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(
            os.environ.get("LAVFI_CC_WEEK5_FFMPEG", str(default_ffmpeg))
        ),
        help="pinned FFmpeg executable containing the fused filter",
    )
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--duration", type=_positive_float, default=30.0)
    parser.add_argument("--samples", type=_positive_int, default=5)
    parser.add_argument("--warmups", type=_nonnegative_int, default=1)
    parser.add_argument("--filter-threads", type=_positive_int, default=10)
    parser.add_argument("--encoder-threads", type=_positive_int, default=4)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / f"{timestamp}-real-video",
    )
    arguments = parser.parse_args()
    if arguments.start < 0:
        parser.error("--start must not be negative")
    return arguments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    case: str,
    kind: str,
    index: int,
    arguments: argparse.Namespace,
    result_dir: Path,
) -> dict[str, str]:
    output = result_dir / f"current-{case}.mp4"
    output.unlink(missing_ok=True)
    ffmpeg_arguments = [
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-benchmark",
        "-ss",
        f"{arguments.start:g}",
        "-t",
        f"{arguments.duration:g}",
        "-i",
        str(arguments.input),
        "-map",
        "0:v:0",
        "-an",
        "-filter_threads",
        str(arguments.filter_threads),
        "-vf",
        CHAIN,
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-threads",
        str(arguments.encoder_threads),
        "-y",
        str(output),
    ]
    command = [str(arguments.ffmpeg), *ffmpeg_arguments]
    if case == "fused":
        command = [
            sys.executable,
            "-m",
            "lavfi_cc",
            "run",
            "--ffmpeg",
            str(arguments.ffmpeg),
            "--require-fusion",
            "--",
            *ffmpeg_arguments,
        ]

    stem = f"{kind}-{case}-{index}"
    (result_dir / f"{stem}.command").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    (result_dir / f"{stem}.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout)[-2000:]
        raise RuntimeError(
            f"{case} {kind} {index} failed with status "
            f"{process.returncode}:\n{detail}"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{case} {kind} {index} did not produce an MP4")

    row = {
        "kind": kind,
        "case": case,
        "index": str(index),
        "wall_seconds": f"{wall_seconds:.6f}",
        "output_bytes": str(output.stat().st_size),
        "sha256": _sha256(output),
        "log_file": f"{stem}.log",
        "command_file": f"{stem}.command",
    }
    output.unlink()
    print(
        f"{kind} {case} {index}: {wall_seconds:.3f} s, "
        f"{row['output_bytes']} bytes, {row['sha256'][:12]}"
    )
    return row


def _write_metadata(
    arguments: argparse.Namespace, result_dir: Path, input_digest: str
) -> None:
    ffmpeg_version = subprocess.run(
        [str(arguments.ffmpeg), "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    ).stdout
    clang = os.environ.get("LAVFI_CC_CLANG", "clang")
    clang_version = subprocess.run(
        [clang, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    ).stdout
    metadata = (
        f"utc_started={datetime.now(timezone.utc).isoformat()}\n"
        f"host_os={platform.system()}\n"
        f"host_release={platform.release()}\n"
        f"host_arch={platform.machine()}\n"
        f"input={arguments.input}\n"
        f"input_sha256={input_digest}\n"
        f"ffmpeg={arguments.ffmpeg}\n"
        f"start_seconds={arguments.start:g}\n"
        f"duration_seconds={arguments.duration:g}\n"
        f"samples={arguments.samples}\n"
        f"warmups={arguments.warmups}\n"
        f"filter_threads={arguments.filter_threads}\n"
        f"encoder_threads={arguments.encoder_threads}\n"
        f"filtergraph={CHAIN}\n\n"
        f"{clang_version}\n{ffmpeg_version}"
    )
    (result_dir / "metadata.txt").write_text(metadata, encoding="utf-8")


def _write_results(rows: list[dict[str, str]], result_dir: Path) -> bool:
    fields = (
        "kind",
        "case",
        "index",
        "wall_seconds",
        "output_bytes",
        "sha256",
        "log_file",
        "command_file",
    )
    with (result_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    recorded = [row for row in rows if row["kind"] == "recorded"]
    baseline = [
        float(row["wall_seconds"])
        for row in recorded
        if row["case"] == "baseline"
    ]
    fused = [
        float(row["wall_seconds"])
        for row in recorded
        if row["case"] == "fused"
    ]
    baseline_median = statistics.median(baseline)
    fused_median = statistics.median(fused)
    speedup = baseline_median / fused_median
    digests = {row["sha256"] for row in recorded}
    sizes = {row["output_bytes"] for row in recorded}
    exact = len(digests) == 1 and len(sizes) == 1
    summary = (
        "# Real-video benchmark summary\n\n"
        "| case | samples | median wall (s) |\n"
        "|---|---:|---:|\n"
        f"| baseline | {len(baseline)} | {baseline_median:.3f} |\n"
        f"| fused | {len(fused)} | {fused_median:.3f} |\n\n"
        f"Baseline / fused speedup: **{speedup:.3f}x**\n\n"
        f"Recorded outputs byte-exact: **{'yes' if exact else 'no'}**\n"
    )
    (result_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return exact


def main() -> int:
    arguments = _arguments()
    arguments.input = arguments.input.expanduser().resolve()
    arguments.ffmpeg = arguments.ffmpeg.expanduser().resolve()
    result_dir = arguments.result_dir.expanduser().resolve()
    if not arguments.input.is_file():
        raise SystemExit(f"benchmark input not found: {arguments.input}")
    if not arguments.ffmpeg.is_file() or not os.access(arguments.ffmpeg, os.X_OK):
        raise SystemExit(f"patched FFmpeg is not executable: {arguments.ffmpeg}")
    result_dir.mkdir(parents=True, exist_ok=True)
    input_digest = _sha256(arguments.input)
    _write_metadata(arguments, result_dir, input_digest)

    rows: list[dict[str, str]] = []
    try:
        for index in range(1, arguments.warmups + 1):
            rows.append(_run("baseline", "warmup", index, arguments, result_dir))
            rows.append(_run("fused", "warmup", index, arguments, result_dir))
        for index in range(1, arguments.samples + 1):
            order = ("baseline", "fused") if index % 2 else ("fused", "baseline")
            for case in order:
                rows.append(_run(case, "recorded", index, arguments, result_dir))
    except RuntimeError as error:
        print(f"real-video benchmark: {error}", file=sys.stderr)
        if rows:
            with (result_dir / "partial-runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return 1

    if not _write_results(rows, result_dir):
        print("real-video benchmark: baseline and fused MP4s differ", file=sys.stderr)
        return 1
    print(f"results: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
