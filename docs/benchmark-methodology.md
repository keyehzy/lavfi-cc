# Week 1 benchmark methodology

Week 1 measures whether repeated pointwise RGBA passes create enough overhead
to justify a fused kernel. It does not claim the final compiler speedup.

## Fixed environment

- FFmpeg tag `n8.1.2`, commit
  `38b88335f99e76ed89ff3c93f877fdefce736c13`.
- Synthetic `testsrc2` at 60 fps, converted to RGBA before the candidate chain
  and constrained to RGBA after it.
- Raw/synthetic input and the null muxer remove decode, encode, and storage
  noise.
- `-filter_threads 1` and the host logical-CPU count are separate treatments.
- macOS arm64 results are directional. Native Linux x86-64 results and `perf`
  counters remain required for the project success criteria.

The harness records the complete command, FFmpeg build configuration, host
identity, warm-up policy, every raw log, and one CSV row per process. FFmpeg's
reported user, system, and real times produce CPU time and frames per second.
On Darwin, FFmpeg 8.1.2 labels `ru_maxrss` as KiB even though Darwin supplies
bytes; the harness divides that value by 1024 before recording `maxrss_kib`.

## Two phases

Screen all ten candidate chains at 1080p and 4K:

```sh
./benchmarks/harness/run.sh screen
```

Screen mode uses one discarded warm-up, one recorded run, and 60 frames by
default. It identifies broken candidates and shows stage-count scaling cheaply.

After screening, run at least five recorded warm samples for the control and
the selected 1-, 2-, 4-, and 8-stage representatives:

```sh
CHAIN_IDS=control_rgba,single_lut,pair_negate_lut,four_balanced,eight_balanced \
  BENCH_FRAMES=120 ./benchmarks/harness/run.sh full
```

Full mode defaults to 300 frames; `BENCH_FRAMES=120` is an allowed Week 1
directional run length on this host. The five-run median, not the best sample,
is used for conclusions. Final performance acceptance should return to 300 or
more frames after the hand-fused filter exists.

## Go/no-go interpretation

The gate passes directionally when a representative four-stage chain is
materially slower than a one-stage chain at 4K and the cost continues to grow
with stage count. This establishes removable full-frame-pass overhead, but it
does not prove the 1.5x fusion target. The hand-fused two-stage experiment is
the next decisive check.

On Linux, add `perf stat` counters for cycles, instructions, LLC misses, and
cache references. On macOS, use Instruments or `xctrace` for profiling and keep
the result explicitly non-authoritative.
