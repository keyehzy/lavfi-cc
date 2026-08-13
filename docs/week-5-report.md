# Week 5 report: FFmpeg integration

Date: 2026-08-13

Decision: **Week 5 exit gate passed on the primary macOS arm64 development
oracle.** A normal FFmpeg invocation containing one supported bounded RGBA8
region is compiled, rewritten to one dynamic `fused` AVFilter, and produces
byte-identical raw frames. The same patch and test entry points target Linux
x86-64; that platform remains the authoritative release and performance host.

## Dynamic AVFilter

`ffmpeg-patch/vf_fused.c` implements the generated-kernel ABI inside the pinned
FFmpeg `n8.1.2` fork. Its options carry an absolute kernel path, its private
trusted root, the source plan hash, and the IR's color-side-data effect.

During initialization the filter:

- resolves the kernel and trusted root to canonical absolute paths;
- requires a current-user-owned, non-group/world-writable root and a regular,
  current-user-owned, non-world-writable kernel directly beneath it;
- loads with immediate local symbol resolution;
- checks ABI version 1, the RGBA8 pixel-format identifier, plan hash, and the
  process-function pointer.

Each frame receives a separately allocated FFmpeg output buffer. The filter
copies frame properties, including timestamps, color fields, metadata, and side
data, removes color-dependent side data only when recorded in the IR, and calls
the unchanged stride-aware kernel ABI through FFmpeg row slices. The filter
accepts only `AV_PIX_FMT_RGBA` and advertises slice threading.

The original hard-coded Week 1 experiment remains available separately as
`vf_fused_week1.c`.

## Wrapper behavior

`lavfi-cc run [wrapper options] -- [FFmpeg arguments]` accepts exactly one
separate `-vf` or `-filter:v` value. The existing fail-closed frontend selects
the bounded region. The wrapper then:

1. creates a mode-0700 per-run directory under `.build/week5`;
2. compiles and independently ABI-validates the generated library;
3. replaces only the supported region, preserving its explicit format guards
   and all opaque filters outside it;
4. preflights the dynamic filter with a synthetic RGBA frame;
5. executes the user's FFmpeg command while the temporary library remains
   alive.

Unsupported or ambiguous commands, compiler failures, and preflight/load
failures run the untouched FFmpeg command by default. `--require-fusion`
instead returns a nonzero status without executing the original command. Once
preflight succeeds, errors from the user's actual FFmpeg command are returned
normally and are not misclassified as fusion failures.

Persistent reuse is deliberately not introduced here; content-addressed cache
keys, atomic publication, corruption recovery, and concurrent compilation are
the Week 6 milestone.

## Verification

Build and test commands:

```sh
./scripts/build-ffmpeg-week5.sh
./scripts/test-week5.sh
```

The pinned fork built successfully with Apple Clang 17. The 68-test suite
passed. Week 5-specific coverage includes:

- command discovery, ambiguity rejection, quoting, and region-only rewriting;
- default fallback and strict-mode behavior for unsupported graphs and compiler
  failures;
- one-, two-, and four-stage native chains at widths 1, 17, and 65, including
  alpha operations, LUTs, levels, and mixers under four FFmpeg filter threads;
- exact comparison of fused and ordinary FFmpeg raw RGBA bytes;
- preservation of timestamps, duration, color fields, and frame metadata;
- rejection of world-writable kernels and plan-hash mismatches;
- all prior parser, IR, interpreter, native-loader, stride, corpus, and pinned
  FFmpeg differential tests.

No byte differences were observed. This satisfies the Week 5 exit condition:
a supported normal-looking FFmpeg command runs through one compiled filter with
bit-exact output.
