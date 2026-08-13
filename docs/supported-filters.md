# Supported-filter semantics for RGBA8

This inventory is derived from FFmpeg `n8.1.2`, commit
`38b88335f99e76ed89ff3c93f877fdefce736c13`. It is the semantic contract for
the compiler, not a summary of behavior across arbitrary FFmpeg releases.

A fused region receives and emits one packed 8-bit RGB format. Every filter
boundary produces another 8-bit value per component. Those intermediate
clipping and quantization points must remain even when the frame pass is fused.
Runtime commands, timeline-dependent options, non-finite values, and format
changes inside the region are rejected.

## Pixel layouts

The accepted layouts are the packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`,
and `bgr24`, and the planar `gbrp` and `gbrap`. They differ only in which byte
holds which component and whether alpha exists, so the IR is
layout-independent and always sees logical RGBA.

One addressing scheme covers both families. A logical channel lives in a plane
at a byte offset within each sample group, and consecutive samples are `step`
bytes apart:

```text
address = plane_base + y * plane_stride + x * step + offset
```

A packed layout has one plane, a step of three or four, and a distinct offset
per channel. A planar layout has one plane per channel, a step of one, and
every offset zero. `lavfi_cc/layouts.py` records both, read out of the pinned
binary by converting one known `0x11223344` RGBA pixel into each format; that
is where `gbrp`'s green, blue, red plane order comes from.

No accepted layout is chroma-subsampled, so a kernel's `width` and `height`
describe every plane it touches.

An alpha-less layout loads `a = 0` and stores only R, G, and B. That matches
upstream: the packed `colorchannelmixer` path omits every alpha term when
`have_alpha` is unset, and the other accepted filters treat components
independently, so an alpha lane that is never stored cannot change a stored one.

Two rules follow from the format lists rather than from the pixel maths:

- A region may only be fused in a format that **every** filter in it
  advertises. Otherwise FFmpeg negotiation inserts a conversion inside the
  region and one kernel is no longer equivalent to the filters it replaced.
  `colorlevels` and `colorchannelmixer` accept `0rgb`, `0bgr`, `rgb0`, and
  `bgr0`, which `negate` and `lutrgb` do not, so that family is outside the
  common subset and is not offered.
- A region may only be fused in a format it already works in. A pointwise
  filter is not format-agnostic: pinned `negate` over `testsrc2` does not
  produce the same bytes as `format=rgba,negate`. Fusing a run whose working
  format is unknown or unsupported would change output, so it is refused.

The 9–16-bit RGB formats are advertised by all four filters but are not
implemented. See [`roadmap-status.md`](roadmap-status.md).

All four upstream filters advertise `AVFILTER_FLAG_SLICE_THREADS`. A fused
filter must do the same. These operations are row-local in the accepted subset,
so changing the row partition must not change pixel values.

## `negate`

Source: `libavfilter/vf_negate.c`.

For every selected channel `c`:

```text
out[c] = 255 - in[c]
```

Unselected channels are copied. There is no rounding or clipping step because
the subtraction is exact over an 8-bit input. The default component mask
selects R, G, and B and passes A through.

The preferred explicit alpha spelling is `components=r+g+b+a` (or
`components=a` for alpha alone). In this pinned revision,
`negate=negate_alpha=1` still passes alpha through for packed RGBA while
negating RGB: the legacy option changes the plane mask, but packed RGB uses the
separate component mask. The compiler must reproduce that pinned behavior if it
accepts the option; it must not infer the behavior suggested by the option name.

An explicit component mask is validated against the format at configuration
time, and a request for a component the format lacks fails the graph outright
("Requested components not available"). The default mask skips that check. The
compiler therefore rejects `components=…a` on `rgb24`, `bgr24`, and `gbrp`
rather than treating it as a no-op, so it accepts exactly the graphs upstream
accepts.

`negate_alpha` sets a **plane** mask, and that mask is only ignored for packed
RGB, which uses its separate component mask instead. Planar RGB obeys it, so
the same option means different things depending on the layout. Over one
`R=0x11 G=0x22 B=0x33 A=0x44` pixel in this pinned revision:

```text
rgba   negate                -> ee dd cc 44     alpha unchanged
rgba   negate=negate_alpha=1 -> ee dd cc 44     alpha unchanged
gbrap  negate                -> dd cc ee 44     alpha unchanged
gbrap  negate=negate_alpha=1 -> dd cc ee bb     alpha negated
```

The compiler accepts the option on packed layouts, where it provably has no
alpha effect, and on `gbrp`, which has no alpha plane. It rejects it on
`gbrap`, where honouring it would require the lowering to be layout-aware;
`components=r+g+b+a` states that intent unambiguously and is accepted.

Accepted MVP constraints:

- Only the `r`, `g`, `b`, and `a` component flags are meaningful at the RGBA
  boundary.
- Parameters are fixed before the first frame. Runtime commands are rejected.

## `lutrgb`

Source: `libavfilter/vf_lut.c`.

At graph configuration, FFmpeg parses one expression for each channel and
evaluates it for integer `val` entries. For RGBA8, `minval=0`, `maxval=255`,
`clipval=clamp(val, 0, 255)`, and `negval=255-clipval`. `w` and `h` are the
fixed input dimensions. Unspecified expressions default to `clipval`, including
alpha.

Each table entry is quantized as:

```text
table[c][val] = clamp((int) expression_result, 0, 255)
out[c] = table[c][in[c]]
```

The C conversion to `int` truncates toward zero; it does not round to nearest.
The result is then saturated. A NaN makes graph configuration fail. The MVP
also rejects non-finite results and any expression outside its documented
parser subset rather than trying to reinterpret it.

The entire table is a compile-time constant for fixed dimensions and static
expressions. A `lutrgb` stage is therefore an explicit `Lut8` followed by an
RGBA8 quantization boundary. Adjacent LUT stages may be composed by table
lookup, which preserves that boundary exactly.

Upstream removes color-dependent frame side data after applying this filter.
A fused chain containing `lutrgb` must preserve that observable metadata
behavior even though its pixel kernel is otherwise local.

## `colorlevels` with `preserve=none`

Source: `libavfilter/vf_colorlevels.c`.

For channel `c`, FFmpeg first quantizes the configured points with `lrint`:

```text
imin = lrint(input_min  * 255)
imax = lrint(input_max  * 255)
omin = lrint(output_min * 255)
omax = lrint(output_max * 255)
coeff = (float) ((omax - omin) / (double) (imax - imin))
```

With the default round-to-nearest floating-point environment, `lrint` resolves
halfway cases using ties-to-even. The coefficient is stored as `float`. The
packed 8-bit pixel path is written upstream as:

```text
value = (in[c] - imin) * coeff + omin
out[c] = clamp((int) value, 0, 255)
```

The pinned Apple-Clang 17 arm64 build contracts the multiply and add to one
binary32 `fmla`, including in its vectorized packed-RGBA loop. IR v2 therefore
records `levels_f32_fma`: compute the exact product-plus-offset and round once
to binary32, then truncate toward zero and saturate. This distinction is
observable for edge configurations; a reversed red range produced 25 under
the oracle where separately rounding the product produced 26 on Linux. The
frontend rejects a channel as `target_sensitive_levels` when those two legal
evaluation modes differ after byte quantization. Accepted mappings therefore
remain portable while IR retains explicit single-rounding semantics. R, G, B,
and (when present) A use independent ranges. Default points are
`input_min=output_min=0` and `input_max=output_max=1`.

Negative input points have special behavior: after point quantization, a
negative `imin` or `imax` is replaced by the observed per-frame channel minimum
or maximum. That is frame-global and not a static pixel-local operation. The
MVP rejects negative input points. It also rejects a quantized `imax == imin`,
which creates a degenerate coefficient, and accepts only `preserve=none`.
Reversed but non-equal endpoints remain eligible when their byte mapping is
independent of multiply-add contraction.

## `colorchannelmixer` with `pc=none`

Source: `libavfilter/vf_colorchannelmixer.c` and
`libavfilter/colorchannelmixer_template.c`.

The 16 coefficients form four output rows (`rr` through `ra`, `gr` through
`ga`, `br` through `ba`, and `ar` through `aa`) over the RGBA input. Each
coefficient is in `[-2, 2]`; defaults form the identity matrix.

For RGBA8, FFmpeg does not evaluate a floating-point dot product per pixel. It
first builds 16 contribution tables:

```text
term[out][in][v] = lrint(v * coefficient[out][in])
```

Then each output is the saturated integer sum of four independently rounded
terms:

```text
out[o] = clamp(term[o][R][r] + term[o][G][g]
             + term[o][B][b] + term[o][A][a], 0, 255)
```

This order is essential: rounding each product and then summing is not generally
equivalent to rounding one matrix dot product. Alpha is transformed by the
fourth row just like RGB. With `pc=none`, preserve amount `pa` is irrelevant.
All other preserve modes are excluded from the MVP.

## Cross-stage and platform rules

- Each stage reads the previous stage's four quantized bytes, even inside a
  fused loop.
- Generated floating-point code uses `-fno-fast-math -ffp-contract=off`; the
  explicit `levels_f32_fma` operation must be implemented with an explicit
  correctly-rounded FMA or a materialized 256-entry table rather than relying
  on ambient compiler contraction.
- The process must use the normal round-to-nearest environment when matching
  upstream `lrint` behavior.
- The same pinned FFmpeg binary is the oracle on each host. Golden frames are
  namespaced by OS, architecture, and FFmpeg revision rather than assumed to be
  cross-platform.
- Frame properties are copied when an upstream filter allocates a new frame.
  `lutrgb` additionally removes color-dependent side data; the other accepted
  filters do not.
