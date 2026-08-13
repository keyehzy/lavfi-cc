#!/usr/bin/env python3
"""Summarize recorded FFmpeg benchmark samples without third-party packages."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUNS.csv", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    groups: dict[tuple[str, int, str, int], list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "recorded" or row["exit_status"] != "0":
                continue
            key = (
                row["chain_id"],
                int(row["stage_count"]),
                row["resolution"],
                int(row["filter_threads"]),
            )
            groups[key].append(row)

    controls: dict[tuple[str, int], float] = {}
    medians: dict[tuple[str, int, str, int], tuple[int, float, float, float, float]] = {}
    for key, rows in groups.items():
        wall = statistics.median(float(row["wall_seconds"]) for row in rows)
        fps = statistics.median(float(row["fps"]) for row in rows)
        cpu = statistics.median(
            float(row["user_seconds"]) + float(row["system_seconds"]) for row in rows
        )
        rss = statistics.median(float(row["maxrss_kib"]) for row in rows)
        medians[key] = (len(rows), wall, fps, cpu, rss)
        if key[0] == "control_rgba":
            controls[(key[2], key[3])] = wall

    summary_csv = source.with_name("summary.csv")
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "chain_id",
                "stage_count",
                "resolution",
                "filter_threads",
                "samples",
                "median_wall_seconds",
                "median_fps",
                "median_cpu_seconds",
                "median_maxrss_kib",
                "wall_vs_control",
            ]
        )
        for key in sorted(medians, key=lambda item: (item[2], item[3], item[1], item[0])):
            samples, wall, fps, cpu, rss = medians[key]
            control = controls.get((key[2], key[3]))
            slowdown = wall / control if control and control > 0 else None
            writer.writerow(
                [
                    *key,
                    samples,
                    f"{wall:.6f}",
                    f"{fps:.3f}",
                    f"{cpu:.6f}",
                    f"{rss:.0f}",
                    f"{slowdown:.3f}" if slowdown is not None else "",
                ]
            )

    markdown = source.with_name("summary.md")
    with markdown.open("w") as handle:
        handle.write("# Benchmark summary\n\n")
        handle.write(
            "Only recorded runs with a zero exit status are included; warm-ups remain in `runs.csv`.\n\n"
        )
        handle.write(
            "| chain | stages | resolution | threads | n | median wall (s) | median fps | wall / control |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for key in sorted(medians, key=lambda item: (item[2], item[3], item[1], item[0])):
            samples, wall, fps, _cpu, _rss = medians[key]
            control = controls.get((key[2], key[3]))
            slowdown = f"{wall / control:.2f}x" if control and control > 0 else "n/a"
            handle.write(
                f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {samples} "
                f"| {wall:.3f} | {fps:.1f} | {slowdown} |\n"
            )

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
