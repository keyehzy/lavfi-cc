# lavfi-cc

`lavfi-cc` is an experimental compiler for fusing compatible FFmpeg video
filters into one native RGBA8 kernel. The repository has completed the Week 4
C-code-generation milestone described in
[`ffmpeg-filter-compiler-mvp.md`](ffmpeg-filter-compiler-mvp.md).

Explain and lower a bounded region with:

```sh
./lavfi-cc explain --vf \
  "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"
```

The command exits with status 0 for an eligible region and 2 for a parse or
eligibility rejection. Add `--json` to obtain the canonical IR, source map,
plan hash, diagnostics, and planned filtergraph rewrite as structured output.
The rewrite remains explanatory until FFmpeg integration arrives in Week 5.

Run the reference interpreter over one or more tightly packed raw RGBA8 frames:

```sh
./lavfi-cc interpret \
  --vf "format=rgba,negate,lutrgb=r=val*1.08+2,format=rgba" \
  --width 1920 --height 1080 \
  --input input.rgba --output output.rgba
```

Omit `--input` or `--output` to use standard input or standard output. Input
must contain only complete `width * height * 4` byte frames. The Python API also
supports padded and negative frame strides through `interpret_into`.

Compile and run the same stream through a temporary checked native kernel:

```sh
./lavfi-cc native \
  --vf "format=rgba,negate,lutrgb=r=val*1.08+2,format=rgba" \
  --width 1920 --height 1080 \
  --input input.rgba --output output.rgba
```

Use `./lavfi-cc compile --vf "..."` to retain readable generated C and a native
`.dylib` or `.so` under `.build/week4`. The loader validates the ABI version,
RGBA8 format identifier, and source plan hash before executing a kernel.
Persistent content-addressed caching remains a Week 6 milestone.

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

Week 3's scalar interpreter, quantization findings, and zero-difference oracle
matrix are in [`docs/week-3-report.md`](docs/week-3-report.md). Run its suite
with:

```sh
./scripts/test-week3.sh
```

Week 4's optimization passes, generated-C ABI, native correctness matrix, and
microbenchmark are in [`docs/week-4-report.md`](docs/week-4-report.md). Run the
full correctness and performance exit gate with:

```sh
./scripts/test-week4.sh
```

Set `LAVFI_CC_FFMPEG=/path/to/ffmpeg` to select another pinned build. The
differential tests skip when neither that variable nor the local Week 1 build
is available.
