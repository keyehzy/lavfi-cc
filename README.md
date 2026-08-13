# lavfi-cc

`lavfi-cc` is an experimental compiler for fusing compatible FFmpeg video
filters into one native RGBA8 kernel. The repository has completed the Week 2
frontend and IR milestone described in
[`ffmpeg-filter-compiler-mvp.md`](ffmpeg-filter-compiler-mvp.md).

Explain and lower a bounded region with:

```sh
./lavfi-cc explain --vf \
  "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"
```

The command exits with status 0 for an eligible region and 2 for a parse or
eligibility rejection. Add `--json` to obtain the canonical IR, source map,
plan hash, diagnostics, and planned filtergraph rewrite as structured output.
The rewrite is explanatory in Week 2; native compilation and execution arrive
in later milestones.

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

Week 2's accepted grammar, IR contract, diagnostics, and results are in
[`docs/week-2-report.md`](docs/week-2-report.md). Run its unit and pinned-FFmpeg
differential suite with:

```sh
./scripts/test-week2.sh
```

Set `LAVFI_CC_FFMPEG=/path/to/ffmpeg` to select another pinned build. The
differential tests skip when neither that variable nor the local Week 1 build
is available.
