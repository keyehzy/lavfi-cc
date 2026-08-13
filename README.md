# lavfi-cc

`lavfi-cc` is an experimental compiler for fusing compatible FFmpeg video
filters into one native kernel over an 8-bit RGB frame. The repository
has completed the Week 5 FFmpeg-integration and Week 6 cache/operational-safety
milestones described in
[`ffmpeg-filter-compiler-mvp.md`](ffmpeg-filter-compiler-mvp.md), plus the reach
work recorded in [`docs/roadmap-status.md`](docs/roadmap-status.md).

Accepted layouts are the packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`, and
`bgr24`, and the planar `gbrp` and `gbrap`. A run is only fused when it already
works in one of them: a pointwise filter produces different bytes in different
pixel formats, so fusing a YUV or negotiation-decided run into an RGB kernel
would change the output. Runs that cannot be fused are reported rather than
guessed at.

Explain and lower a bounded region with:

```sh
./lavfi-cc explain --vf \
  "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"
```

The command exits with status 0 for an eligible region and 2 for a parse or
eligibility rejection. Add `--json` to obtain the canonical IR, source map,
plan hash, diagnostics, and planned filtergraph rewrite as structured output.
The rewrite uses placeholders in `explain`; `run` fills them with a checked
kernel from the private content-addressed cache and its trusted-root path.

Build the pinned FFmpeg fork and run an ordinary command through the fused
filter with:

```sh
./scripts/build-ffmpeg-week5.sh
./lavfi-cc run --require-fusion -- \
  -f lavfi -i "testsrc2=s=1920x1080:r=30" \
  -vf "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p" \
  -frames:v 30 -f null -
```

`run` accepts exactly one separate `-vf` or `-filter:v` argument. Unsupported
graphs, compilation failures, and fused-filter preflight failures run the
original FFmpeg command by default. `--require-fusion` makes those failures
nonzero instead, which is appropriate for tests and benchmarks. Select a
different patched binary with `--ffmpeg PATH` or `LAVFI_CC_FFMPEG`.

## Discovering islands automatically

`--auto-islands` drops the requirement that the fusible run be bracketed by
explicit `format` filters. Every maximal run of supported filters is found
wherever it appears, and each is fused with its own kernel:

```sh
./lavfi-cc explain --auto-islands --vf \
  "format=rgba,negate,lutrgb=r=val*2,crop=64:64,colorlevels=rimin=0.1,negate"
```

The working pixel format is tracked along the chain and survives filters that
provably cannot change it, such as `crop`, `hflip`, and `fps`. `scale` clears
it, because `scale` converts.

## Scanning real filtergraphs

`scan` is analysis only: it never compiles, loads, or runs anything. It accepts
graphs far outside the fusible subset — several chains, link labels, hardware
filters, unreadable options — and reports which runs are fusible today and what
is blocking the rest, ranked by the frame passes each blocker withholds:

```sh
./lavfi-cc scan --file tests/corpus/filtergraphs.txt
./lavfi-cc scan --vf "[0:v]scale=640:360,format=rgba,negate,lutrgb=r=val*2[a]" --json
```

## Building kernels ahead of time

`bundle` compiles every kernel a corpus needs once, at build time, so a
deployment never needs a compiler:

```sh
./lavfi-cc bundle --file graphs.txt --auto-islands --output kernels/
./lavfi-cc run --bundle kernels/ --require-bundle --require-fusion -- \
  -i input.mp4 -vf "format=rgba,negate,lutrgb=r=val*2,format=yuv420p" out.mp4
```

`run` prefers a bundled kernel over compiling one and validates every hit by
checksum, ABI, layout, and plan hash. `--require-bundle` refuses to invoke a
compiler at all. `--emit-only` writes the generated C and the index without
compiling, so another build system can compile the kernels with its own
toolchain. Set `LAVFI_CC_BUNDLE` instead of passing `--bundle` if you prefer.

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

Compile and run the same stream through a cached, checked native kernel:

```sh
./lavfi-cc native \
  --vf "format=rgba,negate,lutrgb=r=val*1.08+2,format=rgba" \
  --width 1920 --height 1080 \
  --input input.rgba --output output.rgba
```

Use `./lavfi-cc compile --vf "..."` to populate or validate the persistent
cache. The command reports `miss`, `hit`, or `rebuilt`; `explain` reports the
same stable cache key and current status. Supplying `--output` and/or `--emit-c`
exports a standalone library/source pair and deliberately bypasses the cache.

Inspect and cap the cache with:

```sh
./lavfi-cc cache list
./lavfi-cc cache prune --max-size 1GiB
```

The default is `~/Library/Caches/lavfi-cc/kernels-v1` on macOS. Linux uses
`$XDG_CACHE_HOME/lavfi-cc/kernels-v1`, falling back to
`~/.cache/lavfi-cc/kernels-v1`. Override it with `--cache-dir PATH` or
`LAVFI_CC_CACHE_DIR`. The cache directory must be owned by
the current user and private (mode 0700); artifacts are mode 0600. Entries are
checksum- and ABI-validated before use. Corrupt or interrupted entries are
rebuilt, concurrent requests for one key compile once, and pruning skips a
kernel while `run` is using it.

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

Week 5's dynamic AVFilter, wrapper rewrite/fallback contract, and integration
matrix are in [`docs/week-5-report.md`](docs/week-5-report.md). Build and run
the complete gate with:

```sh
./scripts/build-ffmpeg-week5.sh
./scripts/test-week5.sh
```

Week 6's cache-key contract, atomic publication and recovery behavior,
concurrency controls, sanitizer result, and warm-cache measurement are in
[`docs/week-6-report.md`](docs/week-6-report.md). Run the complete gate with:

```sh
./scripts/test-week6.sh
```

Island discovery, the widened format set, the scanner, ahead-of-time bundles,
and the reasoning behind what is still missing are in
[`docs/roadmap-status.md`](docs/roadmap-status.md).

The GitHub Actions workflow runs the native suite with Clang 16, 17, and 18 on
Linux, and with Apple Clang and LLVM Clang 18 on macOS. The primary compiler on
each OS also builds both pinned FFmpeg variants, runs all differential and
integration tests, and records filter-only and real-video benchmark artifacts.
Pushes and pull requests use short benchmark samples; the scheduled run and a
manual run with `full_benchmarks` enabled use the full sample counts.

The encoded-video benchmark input is intentionally not committed. CI generates
an encoded MP4 from FFmpeg's deterministic `testsrc2` source, avoiding a
third-party download and any copyrighted repository or cache asset. Reproduce
that baseline-versus-fused measurement with:

```sh
./scripts/build-ffmpeg.sh
./scripts/build-ffmpeg-week5.sh
.build/ffmpeg-macos/bin/ffmpeg -hide_banner -nostdin -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=25" -t 30 -an \
  -c:v mpeg4 -q:v 5 -pix_fmt yuv420p -y video.mp4
python3 scripts/benchmark-real-video.py --start 0
```

On Linux, replace `ffmpeg-macos` with `ffmpeg-linux`. You can also pass any
locally owned or suitably licensed input with `--input` for a content-realistic
benchmark.

The benchmark writes commands, raw logs, metadata, per-run hashes, CSV data,
and a Markdown summary under `benchmarks/results/`. It fails if the recorded
baseline and fused MP4 outputs are not byte-exact.

Set `LAVFI_CC_FFMPEG=/path/to/ffmpeg` to select another pinned build. The
differential tests skip when neither that variable nor the local Week 1 build
is available.
