#!/usr/bin/env python3
"""Authoritative Shutter UI replay benchmark for the application MVP.

The input is an FFmpeg command copied from Shutter's own console. Its output
path must be replaced with ``{output}``, which lets this harness run alternating
baseline and fused treatments without changing any other export setting.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time

from .shutter import (
    EXPECTED_FFMPEG_VERSION,
    ShutterIntegrationError,
    find_filter_complex_argument,
)


ROOT = Path(__file__).resolve().parents[1]
_CODECS = {
    "h264": frozenset({"h264", "libx264"}),
    "prores": frozenset({"prores", "prores_aw", "prores_ks"}),
}


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Replay an actual Shutter UI export command on Linux x86-64"
    )
    parser.add_argument(
        "--command-file",
        required=True,
        type=Path,
        help="one shell-quoted Shutter FFmpeg command containing {output}",
    )
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--codec", required=True, choices=tuple(_CODECS))
    parser.add_argument("--samples", type=_positive_integer, default=5)
    parser.add_argument("--warmups", type=_nonnegative_integer, default=1)
    parser.add_argument("--expected-version", default=EXPECTED_FFMPEG_VERSION)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results" / f"{timestamp}-shutter-ui",
    )
    parser.add_argument(
        "--allow-non-authoritative",
        action="store_true",
        help="permit a development run away from Linux x86-64",
    )
    return parser.parse_args(argv)


def _read_command(path: Path, codec: str) -> tuple[list[str], str]:
    try:
        command = shlex.split(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"could not read the Shutter command: {error}") from error
    if len(command) < 2:
        raise RuntimeError("the command file does not contain an FFmpeg invocation")
    arguments = command[1:]
    find_filter_complex_argument(arguments)

    outputs = [item for item in arguments if "{output}" in item]
    if len(outputs) != 1 or arguments.count(outputs[0]) != 1:
        raise RuntimeError("the Shutter command must contain exactly one {output} token")

    encoder: str | None = None
    for index, item in enumerate(arguments[:-1]):
        if item in {"-c:v", "-codec:v", "-vcodec"}:
            encoder = arguments[index + 1]
    if encoder not in _CODECS[codec]:
        accepted = ", ".join(sorted(_CODECS[codec]))
        raise RuntimeError(
            f"the captured command selects {encoder!r}; --codec {codec} expects {accepted}"
        )
    return arguments, outputs[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_provenance(template: list[str]) -> tuple[Path, str]:
    try:
        input_index = template.index("-i")
    except ValueError as error:
        raise RuntimeError("the captured command has no separate -i input") from error
    if input_index + 1 >= len(template):
        raise RuntimeError("the captured command's -i has no input")
    input_path = Path(template[input_index + 1]).expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(
            "the authoritative benchmark requires a local first input file: "
            f"{input_path}"
        )
    return input_path, _sha256(input_path)


def _decoded_hash(ffmpeg: Path, output: Path) -> str:
    process = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0 or "=" not in process.stdout:
        raise RuntimeError(
            "could not hash decoded frames: "
            + (process.stderr.strip() or f"status {process.returncode}")
        )
    return process.stdout.strip().split("=", 1)[1].lower()


def _materialize_arguments(
    template: list[str], output_token: str, output: Path
) -> list[str]:
    replacement = output_token.replace("{output}", str(output))
    arguments = [replacement if item == output_token else item for item in template]
    if "-y" not in arguments:
        arguments.insert(0, "-y")
    return arguments


def _run_case(
    *,
    case: str,
    kind: str,
    index: int,
    template: list[str],
    output_token: str,
    ffmpeg: Path,
    expected_version: str,
    cache_dir: Path,
    result_dir: Path,
) -> dict[str, str]:
    suffix = Path(output_token.replace("{output}", "sample")).suffix or ".mkv"
    output = result_dir / f"current-{case}{suffix}"
    output.unlink(missing_ok=True)
    ffmpeg_arguments = _materialize_arguments(template, output_token, output)
    command = [str(ffmpeg), *ffmpeg_arguments]
    if case == "fused":
        command = [
            sys.executable,
            "-m",
            "mvp",
            "run",
            "--ffmpeg",
            str(ffmpeg),
            "--cache-dir",
            str(cache_dir),
            "--expected-version",
            expected_version,
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
    wall = time.perf_counter() - started
    (result_dir / f"{stem}.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(
            f"{case} {kind} {index} failed with status {process.returncode}:\n"
            + detail[-2000:]
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{case} {kind} {index} produced no output file")

    row = {
        "kind": kind,
        "case": case,
        "index": str(index),
        "wall_seconds": f"{wall:.6f}",
        "output_bytes": str(output.stat().st_size),
        "encoded_sha256": _sha256(output),
        "decoded_sha256": _decoded_hash(ffmpeg, output),
        "command_file": f"{stem}.command",
        "log_file": f"{stem}.log",
    }
    output.unlink()
    print(
        f"{kind} {case} {index}: {wall:.3f} s, "
        f"encoded={row['encoded_sha256'][:12]} decoded={row['decoded_sha256'][:12]}"
    )
    return row


def _write_report(
    rows: list[dict[str, str]], result_dir: Path, codec: str
) -> bool:
    with (result_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
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
    reduction = (1.0 - fused_median / baseline_median) * 100.0
    encoded_exact = len({row["encoded_sha256"] for row in recorded}) == 1
    decoded_exact = len({row["decoded_sha256"] for row in recorded}) == 1
    report = (
        f"# Shutter UI {codec} benchmark\n\n"
        "| treatment | samples | median wall |\n"
        "|---|---:|---:|\n"
        f"| Shutter FFmpeg chain | {len(baseline)} | {baseline_median:.3f} s |\n"
        f"| lavfi-cc fused | {len(fused)} | {fused_median:.3f} s |\n\n"
        f"Speedup: **{speedup:.3f}x**  \n"
        f"Wall-time reduction: **{reduction:.1f}%**  \n"
        f"Encoded outputs byte-identical: **{'yes' if encoded_exact else 'no'}**  \n"
        f"Decoded frames byte-identical: **{'yes' if decoded_exact else 'no'}**\n"
    )
    (result_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    return encoded_exact and decoded_exact


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    authoritative = (
        platform.system() == "Linux"
        and platform.machine().lower() in {"amd64", "x86_64"}
    )
    if not authoritative and not arguments.allow_non_authoritative:
        print(
            "shutter benchmark: authoritative runs require native Linux x86-64 "
            "(use --allow-non-authoritative only for harness development)",
            file=sys.stderr,
        )
        return 2
    if authoritative and arguments.samples < 5:
        print(
            "shutter benchmark: authoritative runs require at least five samples",
            file=sys.stderr,
        )
        return 2

    ffmpeg = arguments.ffmpeg.expanduser().resolve()
    result_dir = arguments.result_dir.expanduser().resolve()
    if not ffmpeg.is_file():
        print(f"shutter benchmark: FFmpeg not found: {ffmpeg}", file=sys.stderr)
        return 2
    try:
        template, output_token = _read_command(
            arguments.command_file.expanduser().resolve(), arguments.codec
        )
        input_path, input_digest = _input_provenance(template)
    except (RuntimeError, ShutterIntegrationError) as error:
        print(f"shutter benchmark: {error}", file=sys.stderr)
        return 2

    result_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = result_dir / "cache"
    version = subprocess.run(
        [str(ffmpeg), "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    metadata = (
        f"utc_started={datetime.now(timezone.utc).isoformat()}\n"
        f"authoritative={str(authoritative).lower()}\n"
        f"host_os={platform.system()}\n"
        f"host_release={platform.release()}\n"
        f"host_arch={platform.machine()}\n"
        f"codec_gate={arguments.codec}\n"
        f"samples={arguments.samples}\n"
        f"warmups={arguments.warmups}\n"
        f"input={input_path}\n"
        f"input_sha256={input_digest}\n"
        f"expected_ffmpeg_version={arguments.expected_version}\n"
        f"captured_command_file={arguments.command_file.resolve()}\n\n"
        + version
    )
    (result_dir / "metadata.txt").write_text(metadata, encoding="utf-8")
    (result_dir / "captured.command").write_text(
        shlex.join([str(ffmpeg), *template]) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, str]] = []
    try:
        for index in range(1, arguments.warmups + 1):
            for case in ("baseline", "fused"):
                rows.append(
                    _run_case(
                        case=case,
                        kind="warmup",
                        index=index,
                        template=template,
                        output_token=output_token,
                        ffmpeg=ffmpeg,
                        expected_version=arguments.expected_version,
                        cache_dir=cache_dir,
                        result_dir=result_dir,
                    )
                )
        for index in range(1, arguments.samples + 1):
            order = ("baseline", "fused") if index % 2 else ("fused", "baseline")
            for case in order:
                rows.append(
                    _run_case(
                        case=case,
                        kind="recorded",
                        index=index,
                        template=template,
                        output_token=output_token,
                        ffmpeg=ffmpeg,
                        expected_version=arguments.expected_version,
                        cache_dir=cache_dir,
                        result_dir=result_dir,
                    )
                )
    except RuntimeError as error:
        print(f"shutter benchmark: {error}", file=sys.stderr)
        if rows:
            with (result_dir / "partial-runs.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return 1

    exact = _write_report(rows, result_dir, arguments.codec)
    print(f"results: {result_dir}")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
