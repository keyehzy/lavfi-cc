# Week 1 report: baseline and semantic inventory

Date: 2026-08-13

Decision: **GO for Week 2 on the development platform.** Repeated RGBA stages
have large, monotonic cost, and the bit-exact two-stage hand fusion produces a
measurable speedup. This is a directional macOS decision; the authoritative
performance gate remains open until it is repeated on native Linux x86-64.

## Pinned environment

- FFmpeg `n8.1.2`, commit
  `38b88335f99e76ed89ff3c93f877fdefce736c13`.
- macOS 15.7.7, arm64, model `Mac16,1`, 10 logical CPUs, 16 GiB RAM.
- Apple Clang 17.0.0 (`clang-1700.3.19.1`).
- Configure flags: `--cc=/usr/bin/clang --disable-doc --disable-ffplay
  --disable-stripping`, with an isolated prefix under `.build/`.

The complete local build record is
`.build/ffmpeg-macos/build-manifest.txt`. `scripts/build-ffmpeg.sh` recreates it
and verifies the exact revision before building.

## Baseline result

The screen covered the control plus ten candidate chains at 1080p and 4K with
one and ten filter threads. The clean selected run used one discarded warm-up,
five recorded runs, and 120 frames per process. Values below are five-run
median frames per second at 4K; higher is better.

| workload | stages | 1 thread | 10 threads |
|---|---:|---:|---:|
| RGBA conversion control | 0 | 458.0 | 442.8 |
| negate | 1 | 152.1 | 357.1 |
| LUT | 1 | 119.5 | 346.8 |
| mixer | 1 | 35.0 | 150.0 |
| negate + LUT | 2 | 74.8 | 266.7 |
| balanced | 4 | 19.3 | 86.5 |
| LUT-heavy | 4 | 39.6 | 160.4 |
| balanced | 8 | 10.0 | 45.2 |
| LUT-heavy | 8 | 20.7 | 88.2 |

The four-stage LUT-heavy chain takes 3.02x the wall time of the single LUT with
one thread and 2.16x with ten threads. The four-stage balanced chain takes
1.81x/1.74x the wall time of the already-expensive single mixer. Doubling four
stages to eight adds another 1.82x to 1.93x. The cost therefore scales with
full-frame stages even after FFmpeg slice-threading is enabled.

Raw commands, logs, samples, normalized peak RSS, and summaries are retained
locally in:

- `benchmarks/results/20260813-screen/`
- `benchmarks/results/20260813-full-clean/`

macOS does not supply the Linux `perf stat` cycle, instruction, and LLC-miss
counters. Those counters remain a required native-Linux follow-up and are not
inferred from wall time.

## Selected workloads

1. `four_lut_heavy`: `negate,lutrgb,negate,lutrgb`. It is the smallest
   representative whose stages can all be lowered to lookup tables and whose
   intermediate byte semantics can be retained exactly by table composition.
2. `four_balanced`: `negate,lutrgb,colorlevels,colorchannelmixer`. It exercises
   the whole initial subset and retains a large absolute filtering cost after
   slice-threading.

The exact parameters are versioned in `benchmarks/chains.tsv` rather than
repeated here.

## Hand-fused decisive experiment

The Week 1 `fused` filter hardcodes the selected two-stage
`negate,lutrgb` candidate. It always creates a distinct output frame, copies
frame properties, removes color-dependent side data like `lutrgb`, preserves
the intermediate RGBA8 value, and uses FFmpeg slice workers.

Correctness passed byte-for-byte for deterministic RGBA inputs at widths 1, 2,
3, 7, 8, 15, 16, 17, 63, 64, 65, 1919, and 1920 with varying odd/even heights.

Performance used the patched build for both the ordinary and fused chains, 300
frames, one warm-up, and five recorded runs:

| resolution | filter threads | baseline wall | fused wall | speedup |
|---|---:|---:|---:|---:|
| 1920x1080 | 1 | 1.005 s | 0.622 s | 1.62x |
| 1920x1080 | 10 | 0.302 s | 0.201 s | 1.50x |
| 3840x2160 | 1 | 3.973 s | 2.461 s | 1.61x |
| 3840x2160 | 10 | 1.067 s | 0.875 s | 1.22x |

The multi-threaded 4K result is below the final four-stage 1.5x target, but this
experiment removes only one of two passes and pays for a distinct output
allocation. The result is strong enough to justify building the general IR and
testing four-stage fusion. It is not evidence that the final performance
criterion has already been met.

Implementation and results are in `ffmpeg-patch/`,
`tests/corpus/compare-hand-fused.sh`, and
`benchmarks/results/20260813-hand-fused/`.

## Semantic findings that constrain Week 2

- `lutrgb` truncates expression results toward zero before saturation; it does
  not round to nearest.
- `colorlevels` quantizes configured points with `lrint`, stores its coefficient
  as `float`, then truncates the pixel result toward zero before saturation.
- Negative `colorlevels` input points can trigger per-frame extrema scans and
  are excluded from the static pixel-local subset.
- `colorchannelmixer` rounds every coefficient/input contribution separately,
  sums those integers, and only then saturates. A conventional floating-point
  matrix dot product would not be bit-exact.
- On packed RGBA in this revision, `negate_alpha=1` does not negate alpha;
  `components=r+g+b+a` does. The frontend must match the pinned implementation,
  not the option name.
- All four filters are slice-threaded. A fused filter that is single-threaded
  would invalidate the opportunity demonstrated here.

The full contract, accepted-subset decisions, alpha behavior, and metadata
effects are documented in `docs/supported-filters.md`.

## Corpus

Eleven one-frame RGBA oracle outputs cover black, white, ramps, alpha, primary
test patterns, deterministic checker data, all four filters, repeated stages,
odd dimensions, and widths around 8/16/64-byte boundaries. Every file has an
exact command, byte count, and SHA-256 digest. The generated corpus is scoped to
`darwin-arm64-38b88335f99e`; hashes are checked in under
`tests/corpus/manifests/`, while raw files and full commands stay in the ignored
generated directory.

## Remaining before an authoritative performance claim

- Repeat the baseline and hand-fused runs on native Linux x86-64.
- Capture cycles, instructions, cache references, and LLC misses with `perf
  stat`.
- Build and measure the selected four-stage fused workloads.
- Keep decode/filter/encode and cold/warm cache performance for the later MVP
  milestones; Week 1 intentionally isolates filtering.
