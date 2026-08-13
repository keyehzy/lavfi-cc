# Supported-filter semantics for 8-bit pixel formats

This inventory is derived from FFmpeg `n8.1.2`, commit
`38b88335f99e76ed89ff3c93f877fdefce736c13`. It is the semantic contract for
the compiler, not a summary of behavior across arbitrary FFmpeg releases.

A fused region receives and emits one 8-bit pixel format. Every filter
boundary produces another 8-bit value per component. Those intermediate
clipping and quantization points must remain even when the frame pass is fused.
Runtime commands, timeline-dependent options, non-finite values, and format
changes inside the region are rejected.

## Pixel layouts

The accepted layouts are the packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`,
and `bgr24`, the planar RGB `gbrp` and `gbrap`, and the planar YUV `yuv444p`,
`yuv422p`, and `yuv420p`. They differ only in which byte holds which component,
whether alpha exists, and — for YUV — at what resolution each component is
stored, so the IR stays layout-independent and always sees four logical
channels. What those channels are *called* does depend on the layout: red,
green, blue, and alpha in an RGB layout, and luma, Cb, and Cr in a YUV one.
Filters name their options per family and upstream refuses a name the format
does not carry, so the lowering is given the layout.

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
is where `gbrp`'s green, blue, red plane order comes from. YUV's luma, Cb, Cr
plane order was read out the same way, with a component-selective `negate`:
`negate=components=u` changes the second plane and nothing else.

### Chroma subsampling

`yuv422p` stores its chroma planes at half width and `yuv420p` at half width
and half height, rounded up exactly as `AV_CEIL_RSHIFT` does upstream — a
`5x3` `yuv420p` frame has `3x2` chroma planes, not `2x1`. A kernel's `width`
and `height` therefore describe plane 0 only, and the kernel derives each other
plane's dimensions itself from the layout it was generated for.

That changes what an operation may be. One chroma sample covers several luma
samples, so there is no single pixel whose luma and chroma an operation could
mix. The rule is not that a subsampled layout accepts only
channel-independent operations, though — that is too strong, and `hue`'s chroma
rotation is the counterexample. What a subsampled layout cannot express is a
mix of channels that have **no sample in common**. Cb and Cr are sampled at
exactly the same positions in every accepted YUV format, so one loop can hold
both.

`PixelLayout.sampling_groups` partitions the stored channels by sampling shift,
each stage declares the channel groups it reads across, and `validate_ir`
requires each stage group to fit inside one sampling group. `lut8` and the
diagonal `levels_f32_fma` read one channel each and fit anywhere;
`chroma_rotate_i32` reads channels 1 and 2 and fits every accepted YUV layout;
`colorchannelmixer`'s matrix and the `expr_f32` expression read all four and
are refused on `yuv422p` and `yuv420p`. `validate_ir` enforces this for every
consumer of the IR, so it cannot be bypassed by building the IR directly.

Code generation and the interpreter follow the same split: a layout whose
planes share the frame's resolution is walked one pixel at a time with all four
channels loaded, and a subsampled layout is walked one sampling group at a time
at that group's own resolution.

An alpha-less layout loads `a = 0` and stores only the three colour channels.
That matches upstream: the packed `colorchannelmixer` path omits every alpha
term when `have_alpha` is unset, and the other accepted filters treat
components independently, so an alpha lane that is never stored cannot change a
stored one.

Two rules follow from the format lists rather than from the pixel maths:

- A region may only be fused in a format that **every** filter in it
  advertises. Otherwise FFmpeg negotiation inserts a conversion inside the
  region and one kernel is no longer equivalent to the filters it replaced.
  `colorlevels`, `colorchannelmixer`, `colorbalance`, `colorcontrast`, and
  `curves` accept `0rgb`, `0bgr`, `rgb0`, and `bgr0`, which `negate` and
  `lutrgb` do not, so that family is outside the common subset and is not
  offered. In the other direction, `negate` is the only accepted filter that
  advertises both families: `lutrgb`, `colorlevels`, `colorchannelmixer`,
  `colorbalance`, `colorcontrast`, and `curves` are RGB-only and `lutyuv`,
  `eq`, and `hue` are YUV-only, so a run mixing the two is refused rather than
  fused.
- A region may only be fused in a format it already works in. A pointwise
  filter is not format-agnostic: pinned `negate` over `testsrc2` does not
  produce the same bytes as `format=rgba,negate`. Fusing a run whose working
  format is unknown or unsupported would change output, so it is refused.

The 9–16-bit formats, the YUVA and YUVJ families, and the remaining
subsampling ratios (`yuv411p`, `yuv410p`, `yuv440p`) are advertised by
`negate` but are not implemented. See [`roadmap-status.md`](roadmap-status.md).

Every accepted upstream filter advertises `AVFILTER_FLAG_SLICE_THREADS`. A
fused filter must do the same. These operations are row-local in the accepted subset,
so changing the row partition must not change pixel values.

On a subsampled layout the partition is not free, though: a slice boundary must
fall on the chroma sampling grid, or two jobs both own the luma rows covering
one chroma row and race to write it. `vf_fused.c` therefore cuts slices in
units of `1 << log2_chroma_h` plane-0 rows, so each subsampled row belongs to
exactly one job and the per-plane `AV_CEIL_RSHIFT` row counts tile exactly.

## `negate`

Source: `libavfilter/vf_negate.c`.

For every selected channel `c`:

```text
out[c] = 255 - in[c]
```

Unselected channels are copied. There is no rounding or clipping step because
the subtraction is exact over an 8-bit input. The default component mask
selects every colour component of whichever family the format belongs to — R,
G, B or Y, U, V — and passes A through.

Note that the subtrahend is 255 on YUV too. The planar path is
`dst[x] = 255 - src[x]` over each selected plane, so `negate` does not clip
luma to the 16–235 studio range or chroma to 16–240 the way `lutyuv`'s
`clipval` and `negval` variables do.

The preferred explicit alpha spelling is `components=r+g+b+a` (or
`components=a` for alpha alone). In this pinned revision,
`negate=negate_alpha=1` still passes alpha through for packed RGBA while
negating RGB: the legacy option changes the plane mask, but packed RGB uses the
separate component mask. The compiler must reproduce that pinned behavior if it
accepts the option; it must not infer the behavior suggested by the option name.

### The component mask is named per family

An explicit component mask is validated against the format at configuration
time, and a request for a component the format lacks fails the graph outright
("Requested components not available"). The default mask skips that check. Two
consequences, both reproduced by the compiler so it accepts exactly the graphs
upstream accepts:

- `components=…a` is rejected on `rgb24`, `bgr24`, `gbrp`, and every accepted
  YUV layout, rather than treated as a no-op.
- The families are mutually exclusive. `components=r+g+b` fails on `yuv420p`
  and `components=y+u+v` fails on `rgba`, both with the same upstream error.

Within a format the mapping is positional and the same for both families: `r`
and `y` select channel 0, `g` and `u` channel 1, `b` and `v` channel 2, and `a`
channel 3. On YUV the mask becomes a plane mask directly, so
`negate=components=u` negates the Cb plane and copies the other two.

### `negate_alpha` is a plane mask

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
alpha effect, and on layouts with no alpha plane for it to select — `gbrp` and
the accepted YUV formats, whose three planes are all selected either way. It
rejects it on `gbrap`, where honouring it would negate alpha;
`components=r+g+b+a` states that intent unambiguously and is accepted.

Accepted MVP constraints:

- Only the component flags the working format carries are meaningful, and the
  compiler is told the layout so it can say which those are.
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

## `curves`

Source: `libavfilter/vf_curves.c`.

Both slice functions are one table lookup per colour channel:

```text
out[c] = graph[c][in[c]]
```

Alpha is copied. `NB_COMP` is three, so `graph[3]` is the master curve rather
than an alpha one and no curve ever applies to alpha. `curves` is therefore
channel-independent and lowers to the same `Lut8` `lutrgb` lowers to.

All of the work is in `config_input`, which turns key points into the tables,
in `double`:

- `parse_points_str` reads a number, steps over exactly one character, reads
  another, and steps again, so the separators are positional rather than
  meaningful. Coordinates outside `[0, 1]` and points that do not strictly
  increase in `(int)(x * 255)` fail configuration. The compiler reproduces the
  control flow but accepts only plain decimal literals, where `av_strtod` would
  also accept SI postfixes — `0.5k` means 500 upstream.
- `interpolate` is a natural cubic spline: a tridiagonal solve, then
  `a + b*xx + c*xx*xx + d*xx*xx*xx` per entry, with constant padding outside
  the first and last key points. `interpolate_pchip` is the monotonic
  alternative, with `pchip_edge_case` derivatives at the ends and a linear
  special case for two points.
- `CLIP` at 8-bit depth is `av_clip_uint8` of a truncating conversion.
- The master curve is composed on top of every colour channel afterwards,
  `graph[i][j] = graph[3][graph[i][j]]`, whenever its point string was set at
  all — by `master`, `m`, `psfile`, or a preset.
- A preset only fills a component the caller left unset. `all` fills the three
  colour components, also only where unset, and never the master.

Accepted constraints:

- `psfile` and `plot` are rejected: one reads a file the graph does not
  contain, the other writes one, and neither is something a kernel does.
- Naming both spellings of one component — `r` and `red` — is rejected rather
  than resolved to whichever upstream parsed last.
- The table depends on whether the compiler fuses a multiply-add. Both
  evaluations are built and compared for all 256 entries; where they agree the
  table is portable, and where they do not the host decides. See the note under
  `colorbalance` below.

## `colorbalance` with `pl=0`

Source: `libavfilter/vf_colorbalance.c`.

The first accepted filter that is not channel-independent. Each colour channel
is scaled into `[0, 1]`, a per-pixel lightness reads all three, and each output
reads its own channel and that lightness:

```text
r = in[R] / 255,  g = in[G] / 255,  b = in[B] / 255
l = FFMAX3(r, g, b) + FFMIN3(r, g, b)
out[c] = clip_uint8(lrintf(get_component(c, l, s[c], m[c], h[c]) * 255))
```

`get_component` adds three separately rounded terms to the channel value and
clamps with `av_clipf`, where each term is the option scaled by a factor that
depends only on `l`:

```c
s *= av_clipf((b - l) * a + 0.5f, 0.f, 1.f) * scale;      /* a = 4.f  */
m *= av_clipf((l - b) * a + 0.5f, 0.f, 1.f)               /* b = .333f */
   * av_clipf((1.f - l - b) * a + 0.5f, 0.f, 1.f) * scale; /* scale = .7f */
h *= av_clipf((l + b - 1) * a + 0.5f, 0.f, 1.f) * scale;
v += s; v += m; v += h;
return av_clipf(v, 0.f, 1.f);
```

Neither a `Lut8` nor a `matrix4x4` expresses this, so it lowers to `expr_f32`,
a straight-line float32 program that reproduces the operation order above
exactly. Alpha is copied rather than computed, so the program does not store
it.

The four multiply-adds all scale by `4.f`. Scaling by a power of two is exact,
so contracting the multiply into the add rounds the same value once either way
and the lowering is the same on every host. `expr.multiply_is_exact` is asked
rather than assumed, with the bound `l` actually reaches.

`pl=1` runs `preservel`, which scales its multiply-adds by a computed
saturation instead. That is not exact, so the byte would depend on the host;
the option is rejected.

Option values are floats in `[-1, 1]` and `av_opt_set` fails outside that
range, so the compiler rejects out-of-range values rather than clamping them.

## `colorcontrast`

Source: `libavfilter/vf_colorcontrast.c`.

Also `expr_f32`, and the filter that forced the IR to describe multiply-add
fusion. Its `PROCESS` macro pushes each channel along three colour axes,
weights the results, then restores the input lightness:

```text
gd = g - (b + r) * .5   bd = b - (r + g) * .5   rd = r - (g + b) * .5
g0 = g + gd*gm   b0 = b - gd*gm   r0 = r - gd*gm      (and two more axes)
ng = av_clipf((g0*gmw + g1*byw + g2*rcw) * scale, 0.f, 255.f)
li = FFMAX3(r,g,b) + FFMIN3(r,g,b)
lo = FFMAX3(nr,ng,nb) + FFMIN3(nr,ng,nb) + FLT_EPSILON
out[c] = clip_uint8((int) lerpf(n[c], n[c] * li / lo, preserve))
```

Channels enter as raw byte values, not scaled into `[0, 1]`, and the clamps are
against 255. The final conversion truncates toward zero. Alpha is never
touched.

Eighteen of these are multiply-adds Clang contracts at `-ffp-contract=on`: nine
axis shifts, six weighted-sum terms, three `lerpf`s. None scales by a power of
two, so the result differs between a host with a fused multiply-add and one
without — 147 bytes out of the 50,331,648 the complete RGB domain covers. The
IR states the eighteen explicitly, as `fma` where the host fuses and as
separate operations where it does not, and the generated kernel is compiled
with `-ffp-contract=off` so nothing else is fused. Because the choice changes
the program, it changes the plan hash, so the two kinds of host never share a
cached kernel.

`lavfi_cc/target.py` decides which host this is from a machine table rather
than by probing a compiler, because the analysis-only scanner must not invoke
one; `LAVFI_CC_FUSED_MULTIPLY_ADD` overrides it, and a test checks the table
against the real toolchain.

The slice loop is `y < slice_end && sum > FLT_EPSILON`, where `sum` is
`gmw + byw + rcw`. A weight sum at or below `FLT_EPSILON` leaves the frame
untouched, whatever the contrast options say, and the three weights default to
zero. The compiler emits a program that stores nothing there, which the
identity pass removes.

## Cross-stage and platform rules

- Each stage reads the previous stage's four quantized bytes, even inside a
  fused loop.
- Generated floating-point code uses `-fno-fast-math -ffp-contract=off`; the
  explicit `levels_f32_fma` operation must be implemented with an explicit
  correctly-rounded FMA or a materialized 256-entry table rather than relying
  on ambient compiler contraction. The same rule is what lets `expr_f32` say
  which multiply-adds are fused: every one it does not name as `fma` is
  guaranteed to round twice.
- Upstream's own bytes are not portable everywhere. `colorcontrast` and some
  `curves` key points depend on whether the build target has a fused
  multiply-add, so the oracle is the pinned binary **on this host**, and a
  kernel that reproduces it is host-specific by the same amount. The plan hash
  carries that difference, so nothing is shared across it.
- The process must use the normal round-to-nearest environment when matching
  upstream `lrint` behavior.
- The same pinned FFmpeg binary is the oracle on each host. Golden frames are
  namespaced by OS, architecture, and FFmpeg revision rather than assumed to be
  cross-platform.
- Frame properties are copied when an upstream filter allocates a new frame.
  `lutrgb` additionally removes color-dependent side data; the other accepted
  filters do not.
