# Week 4 report: generated C and standalone native kernels

Date: 2026-08-13

Decision: **Week 4 exit gate passed on the primary macOS arm64 development
oracle.** Generated kernels are byte-exact against both the Week 3 interpreter
and pinned FFmpeg `n8.1.2` corpus, and the checked four-stage microbenchmark is
faster than the scalar Python interpreter. Native Linux x86-64 remains required
for release verification; these measurements are not the Week 7 FFmpeg
filter-throughput gate.

## Compiler pipeline

`lavfi_cc.passes` implements a semantics-preserving optimization pipeline with
independent identity-elimination and table-fusion switches:

- Identity elimination recognizes identity LUT, levels, and mixer stages while
  preserving metadata effects from the source filters.
- Levels materialization converts each accepted `levels_f32_fma` mapping into
  its exact four byte tables before stage optimization.
- LUT composition replaces each adjacent LUT run with channel-wise table
  composition. Because every input and output is already a byte table entry,
  this preserves each intermediate quantization boundary.
- A per-channel LUT directly before a mixer is folded into the mixer
  contribution tables. The internal `sum_i32_lut_terms` evaluation records
  that the tables are no longer described by the source matrix coefficients.
  Mixer-to-mixer composition remains deliberately forbidden because the
  intermediate saturation is observable.

A second identity check removes identities exposed by composition, such as two
full negations. `--no-lut-composition` disables levels materialization, LUT
composition, and LUT-to-mixer folding together.

The source IR plan hash remains the native ABI identity. The optimized IR has a
separate diagnostic hash, so enabling a semantics-preserving pass cannot cause
the loader to accept a kernel for a different frontend plan.

`lavfi_cc.codegen` emits deterministic C. Remaining LUT operations use four
256-entry byte tables. Mixer data is transposed to
`[input][value][four-output-contributions]`, using Clang four-lane `int16_t`
vectors. A general mixer therefore performs four indexed vector loads instead
of sixteen scalar loads. All-zero input columns allocate no table, while
zero/identity-only columns such as alpha passthrough are constructed directly.
The four vectors are summed before per-channel saturation; the maximum possible
sum is safely inside `int16_t`. One row loop handles tightly packed, padded,
zero, and negative strides.

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

The full suite passes against the local pinned FFmpeg build. Native kernels are
compared with both the interpreter and raw FFmpeg output for:

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
and reports the median of five preallocated-buffer runs. It now reports both an
uncomposed native kernel and the optimized levels/LUT/mixer kernel, and checks
both against the interpreter before timing them.

```text
cold compile and load       304.276 ms
interpreter median          114.826 ms       0.321 MPix/s
native median                 0.112 ms     328.167 MPix/s
native/interpreter speedup  1022.19x
```

After levels/LUT/mixer folding and packed mixer code generation, the same
checked 256x144 benchmark measured 0.071 ms with composition disabled and
0.027 ms with it enabled, a 2.63x kernel speedup. A native-only 1920x1080 run
measured 4.903 ms versus 1.394 ms, or 3.52x. These are local directional
measurements; CI remains the cross-platform performance record.

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
