# lavfi-cc

`lavfi-cc` is an experimental compiler for fusing compatible FFmpeg video
filters into one native RGBA8 kernel. The repository is currently at the Week
1 baseline and semantic-inventory milestone described in
[`ffmpeg-filter-compiler-mvp.md`](ffmpeg-filter-compiler-mvp.md).

The checked-in Week 1 tooling deliberately uses one pinned FFmpeg build as both
the benchmark subject and semantic oracle:

```sh
./scripts/build-ffmpeg.sh
./benchmarks/harness/run.sh screen
./tests/corpus/generate.sh
```

Run a five-sample benchmark for the stage-count representatives with:

```sh
CHAIN_IDS=control_rgba,single_lut,pair_negate_lut,four_balanced,eight_balanced \
  BENCH_FRAMES=120 ./benchmarks/harness/run.sh full
```

Generated builds, source checkouts, benchmark runs, and platform-specific raw
frames are ignored by Git. Each output directory contains its own metadata,
commands, hashes, and summary, so a result can be archived without relying on
ambient machine state.

Week 1 findings and the gate decision are in
[`docs/week-1-report.md`](docs/week-1-report.md). Exact RGBA8 semantics and
accepted-subset constraints are in
[`docs/supported-filters.md`](docs/supported-filters.md).
