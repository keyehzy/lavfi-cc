# FFmpeg Filter Compiler: MVP Plan

Status: proposed  
Working name: `lavfi-cc`  
Target duration: 8 weeks for one engineer  
Primary platform: Linux x86-64 with Clang

## 1. Objective

Build a proof-of-concept compiler that replaces a compatible linear FFmpeg video-filter chain with one native filter kernel.

The MVP must answer one question:

> Does fusing several pixel-local FFmpeg filters into one frame pass produce a worthwhile speedup while preserving FFmpeg's output exactly?

This is a compiler MVP, not a new media framework. FFmpeg remains responsible for demuxing, decoding, format negotiation, scheduling, encoding, and unsupported filters.

## 2. Success criteria

The MVP is successful when all of the following are true:

1. It accepts an ordinary `-vf` chain from the supported subset and emits or runs an equivalent FFmpeg command containing one fused filter.
2. For supported inputs, fused and unfused output is bit-for-bit identical, verified on decoded raw frames rather than encoded output.
3. A chain of four supported filters is at least 1.5x faster in filter-only benchmarks at 4K RGBA, measured as the median of at least five warm runs.
4. At least one realistic decode-filter-encode workload improves end-to-end throughput by 15% or more when filtering is a material part of runtime.
5. A cached kernel adds less than 20 ms of startup overhead. A cold compile should take less than one second on the reference development machine.
6. An unsupported or failed compilation safely falls back to the original FFmpeg filter chain unless strict mode is requested.

Stretch target: 2x filter-only speedup for a six-filter chain.

## 3. MVP scope

### Included

- Video filters only.
- CPU execution only.
- One input and one output video stream.
- One linear `-vf` chain; no branches or cycles.
- Static filter parameters known before the first frame.
- Fixed frame dimensions during a run.
- Packed 8-bit `RGBA` as the internal pixel format.
- Linux x86-64.
- Native kernel generation through an installed Clang toolchain.
- Persistent on-disk kernel cache.
- The following initial filter subset:
  - `negate`, including optional alpha handling.
  - `lutrgb`, evaluated into compile-time channel lookup tables.
  - `colorlevels` with `preserve=none` and static per-channel values.
  - `colorchannelmixer` with `preserve=none` and a static matrix.
- An explicit `format=rgba` boundary surrounding the fused region so both baseline and fused runs perform the same format conversions.

### Excluded

- Audio filters.
- YUV, floating-point, 10-bit, 12-bit, or 16-bit pixel formats.
- GPU execution.
- `-filter_complex`, multi-input filters, overlays, splits, and joins.
- Geometry-changing filters such as scale, crop, pad, and rotate.
- Neighborhood filters such as blur and sharpen.
- Temporal or stateful filters.
- Runtime filter commands, timeline expressions, or parameters that vary by frame.
- Dynamic resolution or pixel-format changes.
- Third-party filters.
- Windows, macOS, ARM, and cross-compilation.
- Running untrusted filtergraphs in a multi-tenant environment.

Unsupported behavior is deliberately rejected or interpreted by normal FFmpeg. It must never be silently approximated.

## 4. User experience

The wrapper preserves normal FFmpeg arguments after `--`:

```sh
lavfi-cc run -- \
  -i input.mp4 \
  -vf "format=rgba,negate,lutrgb=r='clip(val*1.1)':g=val:b=val,colorlevels=rimin=0.05,format=yuv420p" \
  -c:v libx264 output.mp4
```

For the MVP, the complete region between the two format boundaries must be supported. The wrapper rewrites it conceptually as:

```text
format=rgba,<supported filters>,format=yuv420p
```

to:

```text
format=rgba,fused=kernel=/cache/<hash>.so,format=yuv420p
```

Additional commands:

```sh
lavfi-cc explain --vf "..."
lavfi-cc compile --vf "..." --pixel-format rgba
lavfi-cc cache list
lavfi-cc cache prune --max-size 1GiB
```

`explain` reports:

- Parsed filters and parameters.
- Which region is eligible.
- Why a filter is unsupported.
- Applied compiler passes.
- Kernel cache key and cache status.
- The rewritten FFmpeg filter chain.

Default behavior is safe fallback. `--require-fusion` turns any unsupported construct, compiler error, or kernel-load error into a nonzero exit for tests and benchmarks.

## 5. Architecture

The MVP has four components:

```text
FFmpeg command
     |
     v
lavfi-cc frontend -----> eligibility report
     |
     v
typed pixel IR
     |
     v
C kernel generator -> clang -> cached shared library
                                  |
                                  v
patched FFmpeg -> fused AVFilter -> native kernel
```

### 5.1 Frontend

The frontend extracts and parses the `-vf` argument, validates the supported syntax, and lowers eligible filters to the pixel IR. It delegates all unrelated FFmpeg arguments unchanged.

Do not attempt to parse the entire FFmpeg command grammar. The wrapper only needs to identify `-vf`/`-filter:v`, its following argument, and the input/output format boundaries. Ambiguous invocations are rejected with a useful explanation.

FFmpeg filter escaping is subtle. The first implementation may accept a documented subset, but the parser must fail closed rather than reinterpret text differently from FFmpeg. Before release, differential tests must compare the wrapper parser with FFmpeg's interpretation for every accepted input.

### 5.2 Pixel IR

The IR represents a straight-line program over one RGBA pixel:

```text
LoadRGBA8
Lut8(channel tables)
QuantizeRGBA8
Matrix4x4(coefficients, offsets)
QuantizeRGBA8
Lut8(channel tables)
QuantizeRGBA8
StoreRGBA8
```

Required IR properties:

- Explicit channel order and alpha behavior.
- Explicit integer width, rounding, clipping, and quantization points.
- Source locations that identify the originating filter and option.
- No implicit floating-point contraction.
- A stable serialization used as part of the kernel cache key.

Quantization boundaries are semantically important. Removing an intermediate frame allocation does not permit removing the rounding and clipping that the original filter would have performed before the next filter.

### 5.3 Compiler passes

Implement only passes that are easy to validate:

1. Identity elimination.
2. Constant folding of filter options.
3. Composition of adjacent 8-bit lookup tables.
4. Composition of adjacent matrices only when doing so preserves required intermediate quantization; otherwise retain both operations in one kernel.
5. Dead alpha-operation removal when alpha is known to pass through unchanged.
6. Removal of redundant loads and stores between stages.
7. Loop specialization for `RGBA8`, fixed channel order, and known stage count.

Every optimization must have a switch so it can be disabled independently during differential testing.

### 5.4 Native backend

Generate a small C translation unit exporting a versioned ABI:

```c
typedef struct {
    uint32_t abi_version;
    uint32_t pixel_format;
    const char *plan_hash;
    void (*process)(
        uint8_t *dst,
        ptrdiff_t dst_stride,
        const uint8_t *src,
        ptrdiff_t src_stride,
        int width,
        int height);
} LavfiCompiledKernel;
```

Compile it with Clang into a shared library. Start with conservative optimization flags and explicitly disable transformations that could change FFmpeg rounding behavior. Inspect generated assembly and add vectorization hints only after correctness is locked down.

Why generate C for the MVP:

- It avoids building an assembler or bringing LLVM into the FFmpeg process.
- Clang provides optimization, register allocation, and CPU specialization.
- Generated source is easy to inspect when outputs differ.
- It is disposable: a later version can replace this backend with LLVM ORC, Cranelift, or direct machine-code generation without changing the frontend or IR.

### 5.5 FFmpeg integration

Maintain a small FFmpeg fork containing one new video filter, `fused`.

The filter:

- Accepts only `RGBA8` frames in the MVP.
- Loads a generated shared library using an absolute path supplied by `lavfi-cc`.
- Checks ABI version, pixel format, and plan hash before executing it.
- Allocates the output frame through normal FFmpeg APIs.
- Calls the compiled row/frame function using FFmpeg-provided buffers and strides.
- Preserves frame metadata, timestamps, color metadata, and side data exactly as a normal pointwise filter would.
- Refuses world-writable kernel files and unexpected cache locations.

The fork is a proving mechanism, not the final integration strategy. An upstream design would likely require a safer in-process backend and a first-class graph compilation hook.

### 5.6 Kernel cache

Cache keys must include:

- Canonical serialized IR.
- `lavfi-cc` compiler version.
- Kernel ABI version.
- FFmpeg version or compatible ABI identifier.
- Target architecture and relevant CPU features.
- Clang version and code-generation flags.

Write compiled artifacts atomically. Cache directories and files must be private to the current user. A corrupt or mismatched entry is deleted and rebuilt; it is never executed.

## 6. Correctness strategy

FFmpeg itself is the semantic oracle.

For each test case:

1. Generate or decode identical source frames.
2. Run the ordinary FFmpeg chain and write raw RGBA frames.
3. Run the fused chain and write raw RGBA frames.
4. Compare byte-for-byte and report the first frame, pixel, and channel that differs.

The test matrix should cover:

- Widths around vector and cache-line boundaries: 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 1919, 1920, and 3840.
- Odd and even heights.
- Zero and nontrivial input/output strides.
- Black, white, primary colors, grayscale ramps, alpha ramps, random noise, and `testsrc2`.
- Parameter boundary values and invalid values.
- Multiple filters of the same kind.
- Alternating LUT and matrix stages.
- In-place-looking inputs, although the MVP kernel should use distinct source and destination frames.

Use property-based generation for valid filter chains and parameters. Save minimized failing chains as permanent regression tests.

Run the integration suite under AddressSanitizer and UndefinedBehaviorSanitizer. Compilation and cache tests must also cover concurrent processes requesting the same kernel.

## 7. Benchmark plan

### 7.1 Establish the opportunity first

Before writing the compiler, benchmark ordinary FFmpeg chains to verify that intermediate filtering is expensive enough to optimize.

Use raw or synthetic input and a null/raw sink to remove codec noise:

```text
lavfi test source -> format=rgba -> filter chain -> null sink
```

Measure one, two, four, and eight filter stages at 1080p and 4K. Capture:

- Frames per second.
- Wall time and CPU time.
- Cycles and instructions per frame.
- LLC misses.
- Peak resident memory.
- Bytes copied or an estimated memory-bandwidth figure.

Use `ffmpeg -benchmark`, `perf stat`, and a dedicated benchmark harness that records full commands, versions, warmup policy, and raw results.

Go/no-go gate: a four-stage supported baseline must be meaningfully slower than a one-stage equivalent and show evidence of memory traffic or filter execution as the bottleneck. If it does not, revisit the filter subset before building the JIT.

### 7.2 Benchmark tiers

1. **Kernel microbenchmark:** preallocated RGBA frames, no FFmpeg scheduler.
2. **Filter-only FFmpeg benchmark:** synthetic/raw input to null/raw output.
3. **End-to-end benchmark:** representative H.264 and HEVC decode/filter/encode pipelines.

Report cold compile, warm cache, and baseline separately. Do not include cold compilation in steady-state throughput numbers, but report the break-even frame count.

## 8. Milestones

### Week 1: Baseline and semantic inventory

- Build a pinned FFmpeg revision from source.
- Benchmark candidate chains and select two representative winning workloads.
- Document exact rounding, clipping, alpha, and option behavior for each supported filter.
- Produce a corpus of baseline raw-frame outputs.

Exit gate: filtering overhead is large enough to make the success criteria plausible.

### Week 2: Frontend and IR

- Implement the narrow `-vf` parser and canonicalizer.
- Add eligibility checks and `lavfi-cc explain`.
- Lower supported filters into the typed pixel IR.
- Serialize and pretty-print IR.
- Add parser differential tests and IR unit tests.

Exit gate: every accepted chain has deterministic IR; every unsupported chain has a precise rejection reason.

### Week 3: Reference interpreter

- Implement a deliberately simple scalar IR interpreter.
- Run it outside FFmpeg on raw RGBA frames.
- Differential-test against FFmpeg across generated chains and edge cases.
- Lock down quantization and rounding semantics.

Exit gate: zero byte differences across the test corpus.

### Week 4: C code generation

- Generate readable scalar C from the IR.
- Compile and dynamically load kernels in a standalone harness.
- Add ABI and plan-hash checks.
- Implement identity removal and LUT composition.
- Compare generated kernels against the interpreter and FFmpeg oracle.

Exit gate: generated kernels are bit-exact and faster than the interpreter.

### Week 5: FFmpeg integration

- Add the `fused` AVFilter to the pinned FFmpeg fork.
- Preserve frame properties and side data.
- Implement wrapper command rewriting and safe fallback.
- Add end-to-end raw-frame integration tests.

Exit gate: a supported normal-looking FFmpeg command runs through one compiled filter with bit-exact output.

### Week 6: Cache and operational safety

- Implement stable cache keys, atomic writes, file-permission checks, and corruption recovery.
- Handle concurrent compilation of the same plan.
- Add `compile`, `explain`, cache inspection, and strict-mode commands.
- Run sanitizer and failure-injection tests.

Exit gate: warm-cache execution is reliable and stays below the startup-overhead target.

### Week 7: Performance work

- Profile generated kernels.
- Tune loop structure, restrict qualifiers, alignment paths, and compiler flags.
- Add CPU-feature-specific cache variants if justified.
- Inspect vectorization reports and generated assembly.
- Avoid any optimization that breaks exact output.

Exit gate: meet the filter-only performance target on the reference machine.

### Week 8: Evaluation and release

- Run the full correctness and benchmark matrices.
- Measure realistic end-to-end cases and break-even frame counts.
- Document limitations and reproducible benchmark commands.
- Publish the wrapper, FFmpeg patch, test corpus, and raw benchmark results as an experimental release.

Exit gate: success criteria are met, or the project ends with a documented negative result identifying the actual bottleneck.

## 9. Repository layout

```text
lavfi-cc/
  README.md
  docs/
    architecture.md
    supported-filters.md
    benchmark-methodology.md
  compiler/
    frontend/
    ir/
    passes/
    codegen-c/
  runtime/
    kernel-abi.h
    cache/
  ffmpeg-patch/
    vf_fused.c
    series/
  cmd/
    lavfi-cc/
  tests/
    differential/
    property/
    integration/
    corpus/
  benchmarks/
    harness/
    results/
```

Recommended implementation language: Rust for the wrapper/compiler and C for the FFmpeg filter and generated-kernel ABI. Rust provides good parser and property-testing libraries while keeping the compiler executable easy to distribute. It is not linked into FFmpeg for the MVP.

## 10. Major risks and mitigations

### The baseline filters are already too optimized

Mitigation: make the Week 1 benchmark a hard go/no-go gate. Prefer chains whose cost is repeated full-frame memory traffic. Stop early if fusion cannot plausibly matter.

### Bit-exact behavior prevents useful algebraic optimization

Mitigation: preserve intermediate rounding in registers. The primary win is eliminating frame loads, stores, allocation, and scheduler transitions—not reassociating arithmetic.

### Generated C does not vectorize

Mitigation: first validate that memory-pass elimination alone wins. Then use Clang vectorization reports, specialized stage templates, and explicit SIMD only for measured hot paths.

### Parser behavior diverges from FFmpeg

Mitigation: accept a small syntax subset, differential-test it, and fail closed. A later version should integrate after FFmpeg's own graph parser.

### Dynamic library loading creates a security boundary

Mitigation: load only content-addressed files created in a private cache, verify ABI and hashes, reject unsafe permissions, and document the MVP as unsuitable for untrusted multi-tenant inputs. Replace `dlopen` with an in-process JIT before production use.

### Decode or encode dominates real workloads

Mitigation: publish filter-only and end-to-end results separately. The compiler is valuable only for workloads with sufficiently expensive filter chains.

### Maintaining an FFmpeg fork becomes the project

Mitigation: keep the patch to one filter and build-system registration. Treat upstream integration as a post-MVP design problem.

## 11. Decision after the MVP

Proceed to a second phase only if correctness is exact and the performance gate is met.

Choose the next investment from profiling evidence:

- Add planar YUV and higher bit depths if format conversions limit adoption.
- Replace generated C and `dlopen` with an in-process JIT if compilation or deployment is the main friction.
- Add graph-island discovery after FFmpeg parsing if frontend compatibility is the main friction.
- Add GPU shader generation if upload/download elimination dominates.
- Add neighborhood filters and tiled execution if pointwise fusion succeeds but covers too little real usage.
- Add `-filter_complex` DAG support only after linear-chain fusion is stable.

Do not expand filter coverage merely to increase a support count. Select each new filter because it appears in real workloads and enables a measurable fusion win.

## 12. Immediate first tasks

1. Pin an FFmpeg revision and record the build configuration.
2. Create ten candidate baseline chains using the proposed filter subset.
3. Benchmark them at 1080p and 4K with raw output and a null sink.
4. Read the four upstream filter implementations and document their exact per-pixel semantics.
5. Select the smallest two-filter chain that demonstrates an intermediate-memory-pass cost.
6. Implement that chain manually as one experimental C filter before building the general IR.

Task 6 is the cheapest decisive experiment. If a hand-fused implementation cannot beat the baseline convincingly, a compiler will not rescue the idea.

## References

- [FFmpeg filtergraph documentation](https://www.ffmpeg.org/ffmpeg-filters.html)
- [FFmpeg `libavfilter` graph implementation](https://www.ffmpeg.org/doxygen/trunk/avfiltergraph_8c_source.html)
- [FFmpeg source repository](https://github.com/FFmpeg/FFmpeg/tree/master/libavfilter)


## Extras

This machine is already well prepared:

- macOS 15.7.7, Apple Silicon, 10 cores
- Apple Clang 17
- Rust/Cargo installed
- FFmpeg 8.0.1 with all four proposed filters
- The native four-filter smoke test succeeds

A quick test also revealed that FFmpeg slice-threads these filters. The fused filter must use AVFILTER_FLAG_SLICE_THREADS and FFmpeg’s worker execution mechanism;
otherwise it may replace several parallel filters with one single-threaded kernel.

### Recommended platform split

 Area                       macOS arm64                            Linux x86-64
━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Parser, IR, interpreter    Primary development                    CI verification
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 Generated C kernels        .dylib smoke/correctness               .so release target
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 FFmpeg integration         Native development                     Release verification
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 Differential tests         Same-build macOS oracle                Required acceptance oracle
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 Performance                Directional only                       Authoritative gates
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 Profiling                  Instruments, xctrace, /usr/bin/time    perf stat
─────────────────────────  ─────────────────────────────────────  ────────────────────────────
 Cache location             ~/Library/Caches/lavfi-cc              $XDG_CACHE_HOME/lavfi-cc

Do not use Docker/Rosetta/QEMU x86 emulation for the 1.5× or 15% success criteria. It is fine for functional Linux testing, but distorts CPU and memory behavior.

### 1. Pin and build FFmpeg locally

The Homebrew executable is useful for initial experiments, but it cannot contain the new fused filter. Build a pinned source tree without installing it system-wide.
FFmpeg’s official platform notes confirm that the Xcode toolchain is sufficient for a basic Darwin build. FFmpeg macOS build notes (https://ffmpeg.org/platform.html)

I suggest current stable n8.1.2, commit 38b88335f99e76ed89ff3c93f877fdefce736c13; FFmpeg lists 8.1.2 as the latest stable release. FFmpeg downloads
(https://ffmpeg.org/download.html)

project_root=$PWD

git clone https://git.ffmpeg.org/ffmpeg.git "$project_root/.work/ffmpeg"
git -C "$project_root/.work/ffmpeg" checkout \
  38b88335f99e76ed89ff3c93f877fdefce736c13

build_prefix="$project_root/.build/ffmpeg-macos"

cd "$project_root/.work/ffmpeg"
./configure \
  --prefix="$build_prefix" \
  --cc=/usr/bin/clang \
  --disable-doc \
  --disable-ffplay \
  --disable-stripping

make -j"$(sysctl -n hw.logicalcpu)"
make install

Use only this build for source inspection, oracle output, benchmarks, and the eventual filter patch. Record its commit, -buildconf, Clang version, architecture, and
benchmark commands.

### 2. Add a tiny platform abstraction immediately

Keep all OS differences out of the IR and generated kernel ABI:

macOS: clang -O2 -dynamiclib -fPIC ... -o <hash>.dylib
Linux: clang -O2 -shared     -fPIC ... -o <hash>.so

Initially include:

-std=c11 -O2 -fno-fast-math -ffp-contract=off

Use dlopen(..., RTLD_NOW | RTLD_LOCAL) on both platforms. Generated artifacts must be native to the running FFmpeg process; on this machine that means arm64-apple-darwin.

### 3. Amend the MVP plan before implementation

Add these requirements:

- macOS arm64 is a supported development platform, not the reference release platform.
- Correctness is measured against the same pinned FFmpeg build on each platform; golden raw frames should not be assumed portable between macOS and Linux.
- fused must advertise slice threading and divide work by row ranges.
- Benchmark both -filter_threads 1 and the machine’s normal thread count.
- Cache keys include target triple and dynamic-library format.
- macOS profiling substitutes Instruments/xctrace; final counters still come from Linux perf.

The existing kernel ABI can remain unchanged: each worker can call process() with a pointer to its first row and its slice height.

### 4. Run Week 1 natively

Use macOS to build the benchmark harness and establish whether fusion looks promising. For example:

ffmpeg_bin="$PWD/.build/ffmpeg-macos/bin/ffmpeg"

"$ffmpeg_bin" \
  -hide_banner -nostdin -benchmark \
  -f lavfi -i 'testsrc2=size=3840x2160:rate=60' \
  -frames:v 300 \
  -filter_threads 1 \
  -vf 'format=rgba,negate,lutrgb=r=val:g=val:b=val,colorlevels=rimin=0.05,colorchannelmixer=rr=0.9:rg=0.1:gg=0.9:gb=0.1:bb=0.9:br=0.1,format=rgba' \
  -f null -

Repeat with -filter_threads 10, perform one discarded warm-up plus at least five recorded runs, and retain raw results. FFmpeg documents what -benchmark reports here:
FFmpeg benchmark option (https://www.ffmpeg.org/ffmpeg-all.html).

### 5. Make the hand-fused filter the first implementation

Before creating the general Rust frontend:

1. Pick a simple two-stage chain such as negate,lutrgb.
2. Implement it manually in vf_fused.c.
3. Preserve intermediate 8-bit quantization exactly.
4. Run it through FFmpeg slice workers.
5. Compare raw RGBA output byte-for-byte.
6. Benchmark it on macOS.
7. Repeat the same experiment on native Linux x86-64.

That gives the fastest decisive signal. Parser, IR, cache, and generalized code generation should begin only after the hand-fused experiment shows a credible improvement.
