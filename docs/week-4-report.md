# Week 4 report: generated C and standalone native kernels

Date: 2026-08-13

Decision: **Week 4 exit gate passed on the primary macOS arm64 development
oracle.** Generated kernels are byte-exact against both the Week 3 interpreter
and pinned FFmpeg `n8.1.2` corpus, and the checked four-stage microbenchmark is
faster than the scalar Python interpreter. Native Linux x86-64 remains required
for release verification; these measurements are not the Week 7 FFmpeg
filter-throughput gate.

## Compiler pipeline

`lavfi_cc.passes` implements the two independently switchable Week 4 passes:

- Identity elimination recognizes identity LUT, levels, and mixer stages while
  preserving metadata effects from the source filters.
- LUT composition replaces each adjacent LUT run with channel-wise table
  composition. Because every input and output is already a byte table entry,
  this preserves each intermediate quantization boundary. A second identity
  check removes identities exposed by composition, such as two full negations.

The source IR plan hash remains the native ABI identity. The optimized IR has a
separate diagnostic hash, so enabling a semantics-preserving pass cannot cause
the loader to accept a kernel for a different frontend plan.

`lavfi_cc.codegen` emits deterministic, readable scalar C. LUT operations use
four 256-entry byte tables. `levels_f32_fma` is materialized as four byte tables
at generation time, preserving the Week 3 single-rounding result without
depending on ambient compiler contraction. Mixer stages sum their checked
16-bit contribution tables into 32-bit temporaries before saturation. One row
loop handles tightly packed, padded, zero, and negative strides.

The generated translation unit is compiled with:

```text
-std=c11 -O2 -fPIC -fno-fast-math -ffp-contract=off
```

Darwin uses `-dynamiclib` and `.dylib`; Linux uses `-shared` and `.so`.

## ABI and standalone harness

[`runtime/kernel_abi.h`](../runtime/kernel_abi.h) defines ABI version 1, the
RGBA8 pixel-format identifier, source plan hash, and stride-aware process
function. The Python loader uses local, immediate symbol resolution and refuses
an ABI-version, pixel-format, or plan-hash mismatch before exposing the process
function.

Two commands exercise the backend without FFmpeg integration:

```sh
./lavfi-cc native --vf "format=rgba,negate,format=rgba" \
  --width 1920 --height 1080 --input input.rgba --output output.rgba

./lavfi-cc compile --vf "format=rgba,negate,format=rgba"
```

`native` keeps its generated C and library in a temporary directory for one
stream. `compile` retains both under `.build/week4` by default. Neither path is
the persistent trusted cache planned for Week 6.

## Correctness matrix

Run:

```sh
./scripts/test-week4.sh
```

The 55-test suite passed against the local pinned FFmpeg build. Native kernels
were compared with both the interpreter and raw FFmpeg output for:

- All 11 versioned corpus chains.
- Four all-channel parameter-edge chains.
- Fourteen deterministic one- through six-stage chains over the Week 3 width
  boundary matrix.
- Focused native cases covering every operation, adjacent LUTs, consecutive
  mixers, and mixed four-stage programs.
- Tightly packed, padded positive, zero, and negative strides.
- Optimizations enabled and disabled independently, including a composition
  that exposes an identity.
- Deliberately modified libraries with the wrong ABI version or plan hash.

The suite reports the first native or interpreter byte difference by pixel and
RGBA channel. No differences were observed.

## Performance exit gate

Environment: macOS 15.7.7 arm64; Apple Clang 17.0.0; pinned FFmpeg `n8.1.2`.
The checked benchmark uses a deterministic 256x144 RGBA frame and a four-stage
negate/LUT/levels/mixer chain. It performs one correctness/warm-up invocation
and reports the median of five preallocated-buffer runs.

```text
cold compile and load       304.276 ms
interpreter median          114.826 ms       0.321 MPix/s
native median                 0.112 ms     328.167 MPix/s
native/interpreter speedup  1022.19x
```

The cold compile is below the one-second MVP target, and the native kernel is
unambiguously faster than the deliberately scalar Python oracle. The large
ratio should not be read as an expected FFmpeg fusion speedup: eliminating
Python dispatch dominates this comparison. Week 5 will put the ABI behind an
FFmpeg filter, and Week 7 will measure authoritative filter-only performance on
native Linux x86-64.

## Deferred work

This milestone does not install a persistent cache, validate cache-directory
permissions, recover corrupt artifacts, coordinate concurrent compilers, or
load a kernel from the FFmpeg process. Those remain Week 5 and Week 6 tasks as
specified in the MVP plan.
