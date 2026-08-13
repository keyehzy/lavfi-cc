# Week 3 report: scalar reference interpreter

Date: 2026-08-13

Decision: **Week 3 exit gate passed on the primary macOS arm64 development
oracle.** Every versioned corpus chain and every generated edge case compared
so far has zero byte differences against the pinned FFmpeg `n8.1.2` build.
Native Linux x86-64 remains a required CI and release-verification oracle.

## Standalone execution

`lavfi_cc.interpreter` implements a deliberately scalar IR evaluator. It
validates the IR version, pixel format, load/transform/quantize/store shape,
operation parameters, table dimensions, and quantization-mode pairing before
executing a frame. Its public entry points are:

- `interpret_pixel` for one RGBA8 pixel.
- `interpret_rgba8` for one tightly packed frame.
- `interpret_into` for distinct source and destination buffers with positive,
  padded, or negative byte strides and explicit row offsets.

The command-line form reads consecutive packed frames without loading the full
stream into memory:

```sh
./lavfi-cc interpret \
  --vf "format=rgba,negate,lutrgb=r=val*1.08+2,format=rgba" \
  --width 1920 --height 1080 \
  --input input.rgba --output output.rgba
```

Standard input and output are the defaults. A trailing partial frame is an
error, as are aliased interpreter buffers and layouts that address bytes
outside their buffers.

## Locked quantization semantics

The interpreter retains one byte quantization boundary after every source
filter. LUT stages index concrete byte tables. Mixer stages sum four
independently ties-to-even-rounded integer contribution tables before
saturation. Levels stages use the byte-quantized endpoints and binary32
coefficient encoded in IR.

The edge oracle exposed one correction to the Week 2 semantic inventory. The
pinned Apple-Clang 17 arm64 FFmpeg binary emits `fmla` for the levels expression
`(input - input_min) * coefficient + output_min`. Separately rounding the
multiply can differ by one byte. IR v2 makes the single-rounding behavior
explicit as `levels_f32_fma`; the interpreter evaluates the exact small-integer
and binary32 operands in binary64, where they fit exactly, then rounds once to
binary32. Week 4 code generation must use an explicit FMA or a materialized LUT
instead of depending on compiler contraction settings.

## Differential matrix

Run:

```sh
./scripts/test-week3.sh
```

With the local pinned FFmpeg build available, the suite performs raw-frame
comparisons for:

- All 11 checked-in corpus chains, including one-, two-, four-, and eight-stage
  mixes of every supported filter.
- Four 256-pixel channel ramps covering every possible byte value, reversed
  levels, clipping, negative and maximum mixer coefficients, alpha, and LUT
  truncation.
- Fourteen deterministically generated chains of one through six stages across
  widths 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 1919, 1920, and 3840 and odd or
  even heights.
- Zero, padded positive, and negative-stride unit layouts; multi-frame CLI
  streaming; malformed IR; short buffers; partial frames; and preserved
  intermediate quantization.

The differential test reports the first differing pixel and RGBA channel. On
hosts without a pinned build it skips only the FFmpeg oracle cases;
interpreter, layout, frontend, and CLI unit tests continue to run.
