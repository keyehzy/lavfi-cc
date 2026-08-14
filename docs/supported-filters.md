# Supported-filter semantics for the accepted pixel formats

This inventory is derived from FFmpeg `n8.1.2`, commit
`38b88335f99e76ed89ff3c93f877fdefce736c13`. It is the semantic contract for
the compiler, not a summary of behavior across arbitrary FFmpeg releases.

A fused region receives and emits one pixel format. Every filter boundary
produces another integer value per component, of that format's own depth.
Those intermediate clipping and quantization points must remain even when the
frame pass is fused.
Runtime commands, timeline-dependent options, non-finite values, and format
changes inside the region are rejected.

## Pixel layouts

The accepted layouts are the packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`,
and `bgr24`, the planar RGB `gbrp` and `gbrap`, and the planar YUV `yuv444p`,
`yuv422p`, `yuv420p`, `yuv411p`, `yuv410p`, `yuv440p`, `yuvj444p`,
`yuvj422p`, `yuvj420p`, `yuvj440p`, `yuva444p`, `yuva422p`, and `yuva420p` —
together with the deeper members of each family: the packed `rgb48le`,
`rgba64le`,
`bgr48le`, and `bgra64le`; the planar RGB `gbrp9le`, `gbrp10le`, `gbrap10le`,
`gbrp12le`, `gbrap12le`, `gbrp14le`, `gbrp16le`, and `gbrap16le`; and the
planar YUV `yuv{444,422,420}p{9,10,12,14,16}le`, `yuv440p10le`,
`yuva{444,422,420}p10le`, and `yuva{444,422,420}p16le`. They differ only in
which sample holds which component, whether alpha exists, how wide a sample is,
and — for YUV — at what resolution and range each component is stored. The IR
therefore stays layout-independent and always sees four logical channels. What those channels are *called* does
depend on the layout: red,
green, blue, and alpha in an RGB layout, and luma, Cb, and Cr in a YUV one.
Filters name their options per family and upstream refuses a name the format
does not carry, so the lowering is given the layout.

One addressing scheme covers both families and both widths. A logical channel
lives in a plane at a sample offset within each sample group, consecutive
groups are `step` samples apart, and a sample is one byte up to eight bits and
two above:

```text
address = plane_base + y * plane_stride
          + (x * step + offset) * sample_bytes
```

A packed layout has one plane, a step of three or four, and a distinct offset
per channel. A planar layout has one plane per channel, a step of one, and
every offset zero. Counting in samples rather than bytes is what lets one
scheme serve both widths; at eight bits the two counts coincide, which is why
no eight-bit layout's numbers changed when the deeper ones were added.
`lavfi_cc/layouts.py` records both, read out of the pinned
binary by converting one known `0x11223344` RGBA pixel into each format; that
is where `gbrp`'s green, blue, red plane order comes from. YUV's luma, Cb, Cr
plane order was read out the same way, with a component-selective `negate`:
`negate=components=u` changes the second plane and nothing else. `yuva`'s
fourth plane was read out by the same probe, one component at a time:
`negate=components=a` changes the last plane, which is `width * height` bytes
rather than a chroma plane's size.

Only the `le` member of each deep format is a layout. A kernel loads a native
`uint16` rather than byte-swapping, so the generated source carries an
`#error` that fails the build on a big-endian host instead of producing wrong
bytes there.

### The domain a kernel is defined over

A sample of a format with depth *d* means a value in `[0, 2^d - 1]`, and that
is the domain the interpreter and the generated kernels are bit-exact over.
Outside it the accepted filters do not agree with each other about what
happens, so no compiler could match all of them at once: `vf_lut.c` carries a
full 65536-entry table whatever the depth and answers from it, `vf_hue.c`
clamps the sample before indexing, and `vf_curves.c` and
`vf_colorchannelmixer.c` index a `1 << d`-entry table with the raw sample and
read past its end. Every table here is therefore sized to the format's own
domain and indexed through a clamp, which is defined and safe for a malformed
frame without pretending to reproduce an upstream answer that does not exist.
At eight and sixteen bits the table covers every value the sample type can
hold, so no clamp is emitted at all.

This is also why the differential tests feed samples inside the format's
domain: with wider ones the oracle itself is reading past the end of its own
table, and there is nothing to be exact against.

### Chroma subsampling

`yuv422p` stores its chroma planes at half width, `yuv420p` at half width and
half height, `yuv411p` at quarter width, `yuv410p` at quarter width and quarter
height, and `yuv440p`/`yuv440p10le` at half height. Every dimension is rounded
up exactly as `AV_CEIL_RSHIFT` does upstream — a
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
requires each stage group to fit inside one sampling group. A LUT and the
diagonal `levels_f32_fma` read one channel each and fit anywhere;
`chroma_rotate_i32` reads channels 1 and 2 and fits every accepted YUV layout;
`colorchannelmixer`'s matrix and the RGB `expr_f32` expressions read across
the colour channels and are refused on `yuv422p` and `yuv420p`. `validate_ir`
enforces this for every consumer of the IR, so it cannot be bypassed by
building the IR directly.

The `yuva` layouts are where the partition stops being a restatement of "is
this subsampled". Only their chroma planes shrink; alpha keeps the frame's
resolution, exactly as luma does — upstream sets `height[0] = height[3] =
inlink->h` and sizes only planes 1 and 2 with `AV_CEIL_RSHIFT`. So `yuva420p`
partitions into `(luma, alpha)` and `(Cb, Cr)`: two groups whose members are
not adjacent planes, in a layout that subsamples *and* carries alpha at once.
An operation reading luma together with alpha is admissible there for the same
reason `hue`'s rotation is admissible — they share a sample grid — while a
colour matrix reaching across luma and chroma is still refused.

Code generation and the interpreter follow the same split: a layout whose
planes share the frame's resolution is walked one pixel at a time with all four
channels loaded, and a subsampled layout is walked one sampling group at a time
at that group's own resolution. A `yuva420p` kernel therefore contains loops of
two different resolutions, with plane 3 walked at plane 0's row count.

An alpha-less layout loads `a = 0` and stores only the three colour channels.
That matches upstream: the packed `colorchannelmixer` path omits every alpha
term when `have_alpha` is unset, and the other accepted filters treat
components independently, so an alpha lane that is never stored cannot change a
stored one. Where alpha *is* stored, the accepted YUV filters pass it through
rather than compute it: `eq` copies plane 3 outright, `hue` copies it, `negate`
touches it only when asked, and `lutyuv` applies a table whose default
expression is the identity over alpha's full sample range.

Two rules follow from the format lists rather than from the pixel maths:

- A region may only be fused in a format that **every** filter in it
  advertises. Otherwise FFmpeg negotiation inserts a conversion inside the
  region and one kernel is no longer equivalent to the filters it replaced.
  `colorlevels`, `colorchannelmixer`, `colorbalance`, `colorcontrast`,
  `curves`, `vibrance`, `colortemperature`, and `selectivecolor` accept at
  least part of the `0rgb`/`rgb0` family, which is outside the implemented
  layouts and is not offered. In the other direction, `negate` is the only
  accepted filter that advertises both families: the other RGB filters are
  RGB-only and `lutyuv`, `eq`, and `hue` are YUV-only, so a run mixing the two
  is refused rather than fused. `selectivecolor` narrows this once more: it is
  packed-only and therefore splits a planar RGB run.
- A region may only be fused in a format it already works in. A pointwise
  filter is not format-agnostic: pinned `negate` over `testsrc2` does not
  produce the same bytes as `format=rgba,negate`. Fusing a run whose working
  format is unknown or unsupported would change output, so it is refused.

Depth narrows the first rule further, and not uniformly, because each filter
has its own format list rather than a shared one:

| filter | above eight bits |
|---|---|
| `negate` | every accepted deep format, RGB and YUV alike |
| `lutrgb` | planar RGB and `rgb48le`/`rgba64le`, but neither `bgr` order |
| `lutyuv` | every deep planar YUV, including `yuv440p10le`, but alpha-carrying only at 16 bits |
| `hue` | ten bits only, including `yuv440p10le` and the ten-bit `yuva` trio |
| `eq` | nothing: `vf_eq.c` is an 8-bit filter |
| `colorlevels`, `colorchannelmixer`, `colorbalance`, `colorcontrast`, `curves` | every accepted deep RGB format |
| `vibrance`, `colortemperature` | every accepted deep RGB format |
| `selectivecolor` | the four packed 16-bit RGB/BGR layouts only |

Two consequences are worth naming because they are refusals inside one colour
family. `eq` leaves every run above eight bits, so a deep YUV island is built
from `negate`, `lutyuv`, and `hue` alone. And `lutyuv` and `hue` never share an
alpha-carrying deep format — `vf_lut.c` lists the `yuva` formats at sixteen
bits only and `vf_hue.c` at ten only — so `format=yuva420p10le,lutyuv=…,hue=…`
is not one run.

At eight bits the lists differ too. `negate` and `lutyuv` advertise all accepted
YUV layouts; `eq` omits YUVJ and 4:4:0; and `hue` omits YUVJ but includes
`yuv411p`, `yuv410p`, and `yuv440p`.

Every accepted upstream filter advertises `AVFILTER_FLAG_SLICE_THREADS`. A
fused filter must do the same. These operations are row-local in the accepted
subset, so changing the row partition must not change pixel values.

On a subsampled layout the partition is not free, though: a slice boundary must
fall on the chroma sampling grid, or two jobs both own the luma rows covering
one chroma row and race to write it. `vf_fused.c` therefore cuts slices in
units of `1 << log2_chroma_h` plane-0 rows, so each subsampled row belongs to
exactly one job and the per-plane `AV_CEIL_RSHIFT` row counts tile exactly.

## `negate`

Source: `libavfilter/vf_negate.c`.

For every selected channel `c`:

```text
out[c] = max - in[c]        max = (1 << depth) - 1
```

Unselected channels are copied. There is no rounding or clipping step because
the subtraction is exact over an in-range input. The default component mask
selects every colour component of whichever family the format belongs to — R,
G, B or Y, U, V — and passes A through.

Note that the subtrahend is the format's own maximum on YUV too, and that
`negate` is its own filter rather than a `vf_lut.c` entry point. The planar
path is `dst[x] = max - src[x]` over each selected plane, so `negate` does not
clip luma to the 16–235 studio range or chroma to 16–240 the way `lutyuv`'s
`clipval` and `negval` variables do on limited-range YUV — at any depth. YUVJ's
`lutyuv` variables are full-range instead.

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

- `components=…a` is rejected on `rgb24`, `bgr24`, and every alpha-less
  planar RGB or YUV layout, rather than treated as a no-op. It is accepted on
  every layout that does carry alpha, including the four-channel packed,
  `gbrap`, and `yuva` families; that is the whole difference between
  `yuv420p10le` and `yuva420p10le` as far as this option is concerned.
- The families are mutually exclusive, and carrying alpha does not change
  which family a format belongs to. `components=r+g+b` fails on `yuv420p` and
  on `yuva420p`, and `components=y+u+v` fails on `rgba`, all with the same
  upstream error.

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

Planar YUV with alpha behaves like `gbrap` rather than like packed RGB, since
what decides it is `is_packed` and not the family. Over one frame whose four
planes hold `Y=0x11 U=0x22 V=0x33 A=0x44`:

```text
yuv420p  negate=negate_alpha=1 -> ee dd cc        no alpha plane to select
yuva420p negate                -> ee dd cc 44     alpha unchanged
yuva420p negate=negate_alpha=1 -> ee dd cc bb     alpha negated
```

The compiler accepts the option on packed layouts, where it provably has no
alpha effect, and on layouts with no alpha plane for it to select — the
alpha-less planar RGB and YUV formats, whose three planes are all selected
either way. It rejects it on every alpha-carrying planar RGB or YUV layout,
where honouring it would negate alpha. The suggested spelling is in the
layout's own family, since naming the other one is itself a configuration
failure: `components=r+g+b+a` on `gbrap16le` and
`components=y+u+v+a` on `yuva420p16le`.

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
table[c][val] = clamp((int) expression_result, 0, max_a)
out[c] = table[c][in[c]]
```

The C conversion to `int` truncates toward zero; it does not round to nearest.
The result is then saturated. A NaN makes graph configuration fail. The MVP
also rejects non-finite results and any expression outside its documented
parser subset rather than trying to reinterpret it.

`config_props` is a switch over the pixel format with three arms, and the
range it picks is the whole difference between the entry points and between
the depths:

- Limited-range planar YUV, at any depth: luma
  `16 << (d - 8) .. 235 << (d - 8)`, each chroma channel to
  `240 << (d - 8)`, alpha the full `0 .. (1 << d) - 1`. Alpha also supplies
  `max_a`, the bound every component's result is clipped to.
- The deprecated YUVJ aliases: a true `0 .. 255` for luma and both chroma
  components. They fall through the default switch arm in pinned `vf_lut.c`,
  so `clipval` is an identity and `negval` is `255-val`.
- `rgb48le` and `rgba64le`: a true `0 .. 65535`.
- Everything else, including all the planar RGB formats: `0 .. 255 << (d - 8)`.
  So a `gbrp10le` component runs to 1020 rather than 1023, and 1020 is the clip
  bound too. That is upstream's arithmetic, reproduced rather than corrected.

The entire table is a compile-time constant for fixed dimensions and static
expressions. A `lutrgb` stage is therefore an explicit table lookup followed by
a quantization boundary. Adjacent LUT stages may be composed by table
lookup, which preserves that boundary exactly.

Upstream removes color-dependent frame side data after applying this filter.
A fused chain containing `lutrgb` must preserve that observable metadata
behavior even though its pixel kernel is otherwise local.

## `colorlevels` with `preserve=none`

Source: `libavfilter/vf_colorlevels.c`.

For channel `c`, FFmpeg first quantizes the configured points with `lrint`:

```text
imin = lrint(input_min  * scale)
imax = lrint(input_max  * scale)
omin = lrint(output_min * scale)
omax = lrint(output_max * scale)
coeff = (float) ((omax - omin) / (double) (imax - imin))
```

`scale` is `UINT8_MAX` when a sample is one byte and `UINT16_MAX` when it is
two — whatever the actual depth is. So a `gbrp10le` endpoint runs to 65535
while the samples it is compared against stop at 1023, which makes
`rimin=0.05` clip the whole channel there. That is a quirk of
`vf_colorlevels.c`, reproduced rather than fixed.

With the default round-to-nearest floating-point environment, `lrint` resolves
halfway cases using ties-to-even. The coefficient is stored as `float`. The
packed pixel path is written upstream as:

```text
value = (in[c] - imin) * coeff + omin
out[c] = clamp((int) value, 0, (1 << depth) - 1)
```

The pinned Apple-Clang 17 arm64 build contracts the multiply and add to one
binary32 `fmla`, including in its vectorized packed-RGBA loop. The IR therefore
records `levels_f32_fma`: compute the exact product-plus-offset and round once
to binary32, then truncate toward zero and saturate. This distinction is
observable for edge configurations; a reversed red range produced 25 under
the oracle where separately rounding the product produced 26 on Linux.

Whether those two legal evaluation modes differ after quantization is checked
over the format's whole domain. When they agree — the usual case at eight bits
— the operation says nothing about contraction and means the same bytes on
every host. When they part, the operation records `contraction: fused` or
`separate` from the host machine table, exactly as `colorcontrast` does, and
the plan hash carries it so the two kinds of host can never share a kernel.
Stating it rather than refusing it is not a preference: at fourteen and sixteen
bits a 200-sample sweep of ordinary option sets disagrees somewhere in 118 and
179 cases respectively, where at eight, nine, and ten bits it never does. R, G,
B, and (when present) A use independent ranges. Default points are
`input_min=output_min=0` and `input_max=output_max=1`.

Negative input points have special behavior: after point quantization, a
negative `imin` or `imax` is replaced by the observed per-frame channel minimum
or maximum. That is frame-global and not a static pixel-local operation. The
MVP rejects negative input points. It also rejects a quantized `imax == imin`,
which creates a degenerate coefficient, and accepts only `preserve=none`.
Reversed but non-equal endpoints remain eligible, with the host's contraction
recorded when it decides a sample.

## `colorchannelmixer` with `pc=none`

Source: `libavfilter/vf_colorchannelmixer.c` and
`libavfilter/colorchannelmixer_template.c`.

The 16 coefficients form four output rows (`rr` through `ra`, `gr` through
`ga`, `br` through `ba`, and `ar` through `aa`) over the RGBA input. Each
coefficient is in `[-2, 2]`; defaults form the identity matrix.

FFmpeg does not evaluate a floating-point dot product per pixel. It
first builds 16 contribution tables, one entry per value the format can hold:

```text
term[out][in][v] = lrint(v * coefficient[out][in])
```

Then each output is the saturated integer sum of four independently rounded
terms:

```text
out[o] = clamp(term[o][R][r] + term[o][G][g]
             + term[o][B][b] + term[o][A][a], 0, (1 << depth) - 1)
```

This order is essential: rounding each product and then summing is not generally
equivalent to rounding one matrix dot product. Alpha is transformed by the
fourth row just like RGB. With `pc=none`, preserve amount `pa` is irrelevant.
All other preserve modes are excluded from the MVP.

Nothing about that arithmetic changes with depth; only the tables grow, and the
generated vector the four terms are summed in widens from `int16_t` to
`int32_t`, since a term is a sample scaled by up to two.

## `curves`

Source: `libavfilter/vf_curves.c`.

Both slice functions are one table lookup per colour channel:

```text
out[c] = graph[c][in[c]]
```

Alpha is copied. `NB_COMP` is three, so `graph[3]` is the master curve rather
than an alpha one and no curve ever applies to alpha. `curves` is therefore
channel-independent and lowers to the same table `lutrgb` lowers to.

All of the work is in `config_input`, which turns key points into the tables,
in `double`:

- `config_input` sizes every graph at `1 << depth` entries and passes the depth
  to both interpolators as `nbits`, so `scale` — what upstream calls the white
  value — is `lut_size - 1`. A deeper table is the same curve sampled more
  finely, not a different one.
- `parse_points_str` reads a number, steps over exactly one character, reads
  another, and steps again, so the separators are positional rather than
  meaningful. Coordinates outside `[0, 1]` and points that do not strictly
  increase in `(int)(x * scale)` fail configuration — so a pair of key points
  upstream refuses at eight bits it can accept at ten. The compiler reproduces
  the control flow but accepts only plain decimal literals, where `av_strtod`
  would also accept SI postfixes — `0.5k` means 500 upstream.
- `interpolate` is a natural cubic spline: a tridiagonal solve, then
  `a + b*xx + c*xx*xx + d*xx*xx*xx` per entry, with constant padding outside
  the first and last key points. `interpolate_pchip` is the monotonic
  alternative, with `pchip_edge_case` derivatives at the ends and a linear
  special case for two points.
- `CLIP` is `av_clip_uint8` at eight bits and `av_clip_uintp2_c(v, nbits)`
  above, both of a truncating conversion.
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
  evaluations are built and compared over the whole table; where they agree the
  table is portable, and where they do not the host decides. See the note under
  `colorbalance` below. A deeper table has more entries to land on a boundary,
  so this happens more often rather than less.

## `colorbalance` with `pl=0`

Source: `libavfilter/vf_colorbalance.c`.

The first accepted filter that is not channel-independent. Each colour channel
is scaled into `[0, 1]`, a per-pixel lightness reads all three, and each output
reads its own channel and that lightness:

```text
max = (1 << depth) - 1
r = in[R] / max,  g = in[G] / max,  b = in[B] / max
l = FFMAX3(r, g, b) + FFMIN3(r, g, b)
out[c] = clip(lrintf(get_component(c, l, s[c], m[c], h[c]) * max), depth)
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

Neither a table nor a `matrix4x4` expresses this, so it lowers to `expr_f32`,
a straight-line float32 program that reproduces the operation order above
exactly. Alpha is copied rather than computed, so the program does not store
it.

The four multiply-adds all scale by `4.f`. Scaling by a power of two is exact,
so contracting the multiply into the add rounds the same value once either way
and the lowering is the same on every host. `expr.multiply_is_exact` is asked
rather than assumed, with the bound `l` actually reaches — and that bound is
`[0, 2]` at every depth, since each channel is divided by the format's maximum
on the way in and multiplied by it on the way out. `color_balance16_p` is
`color_balance8_p` with `s->max` widened and nothing else, so the exactness
argument carries over unchanged.

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
ng = av_clipf((g0*gmw + g1*byw + g2*rcw) * scale, 0.f, max)
li = FFMAX3(r,g,b) + FFMIN3(r,g,b)
lo = FFMAX3(nr,ng,nb) + FFMIN3(nr,ng,nb) + FLT_EPSILON
out[c] = clip((int) lerpf(n[c], n[c] * li / lo, preserve), depth)
```

Channels enter as raw sample values, not scaled into `[0, 1]`, and the clamps
are against the format's maximum — `PROCESS` takes that bound as a macro
argument, so the deeper slice functions are the same eighteen fusion sites with
`max` widened. The final conversion truncates toward zero. Alpha is never
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

## `vibrance`

Source: `libavfilter/vf_vibrance.c`.

The filter normalizes RGB by the format maximum, measures saturation as the
maximum channel minus the minimum, and forms a configurable luma. Each channel
then gets its own saturation coefficient:

```text
saturation = max(r,g,b) - min(r,g,b)
luma = g*glum + r*rlum + b*blum
ci = 1 + (intensity*balance[i])
             * (1 - alternate*sign(intensity*balance[i])*saturation)
out[i] = clip((int)((luma + (in[i]/max-luma)*ci) * max), depth)
```

`alternate=0`, the default, makes the sign factor negative; `alternate=1`
makes it positive. Options are stored as binary32 and retain upstream's ranges:
intensity `[-2,2]`, balances `[-10,10]`, and luma coefficients `[0,1]`.
Alpha is copied.

On a fused-multiply-add target the pinned compiler contracts the two luma
adds, the two coefficient multiply-adds per channel, and the three final
`lerpf`s. The expression records those eleven written sites explicitly and
uses the source's left-associative luma expression on a non-contracting target.
With zero effective intensity, a fused `lerpf` recovers the input exactly and
is emitted as an identity. The separate `(input-luma)+luma` can lose a low bit,
so the same default invocation remains an expression on a non-contracting host;
the plan states that otherwise surprising upstream difference.

## `colortemperature`

Source: `libavfilter/vf_colortemperature.c`.

`kelvin2rgb` turns the configured 1000–40000 K temperature into three binary32
multipliers with `logf` or `powf`. The lowering calls the host C library's
float entry points, the same library the pinned binary uses, and includes the
three resulting constants in the canonical plan. That makes the otherwise
frame-constant setup calculation explicit rather than evaluating libm for
every pixel in the generated kernel.

The pixel expression multiplies RGB by those constants, blends it with the
input by `mix`, computes input and output lightness as `max + min +
FLT_EPSILON`, and blends the lightness-restored result by `pl`:

```text
n[c] = lerpf(in[c], in[c] * kelvin_rgb[c], mix)
l = (max(in)+min(in)+FLT_EPSILON) / (max(n)+min(n)+FLT_EPSILON)
out[c] = clip((int)lerpf(n[c], n[c] * l, pl), depth)
```

The six `lerpf` sites are explicit fused or separate multiply-adds according
to the host. `mix=0` is an identity and does not consult either libm or the
target table. Alpha is copied. All implemented RGB layouts are advertised.

## `selectivecolor`

Source: `libavfilter/vf_selectivecolor.c`.

This filter advertises packed RGB only: the six implemented eight-bit packed
layouts and all four packed 16-bit RGB/BGR layouts. A `gbrp` run containing it
is therefore split by upstream negotiation and is refused here even though its
per-pixel arithmetic could be described.

Each configured option contains one to four whitespace-separated CMYK
adjustments; omitted fields are zero. Nine ranges are visited in upstream's
fixed order: reds, yellows, greens, cyans, blues, magentas, whites, neutrals,
and blacks. The first six classify ties too — a grey pixel is simultaneously
at every channel maximum and minimum — while the last three use the exact
integer midpoint predicates from the 8- or 16-bit slice. Every active range
computes an integer scale from the pixel's minimum, middle, and maximum.

For output component value `v`, component adjustment `a`, and black adjustment
`k`, one range contributes:

```text
res = (-1 - a) * k - a
if correction_method == relative: res *= 1 - v/max
term = lrintf(clip(res, -v/max, 1-v/max) * range_scale)
```

Those `term` values are added as integers only after each range has rounded;
rounding the combined adjustment once is observably different. `expr_f32`
therefore includes exact predicates, `abs`/`floor` for the integer range
scales, and an intermediate `lrintf` operation. Their results remain exactly
representable binary32 integers at these bounds. The multiply-add in `res` is
stated as fused or separate for the host, and the final channel sum truncates
and saturates like `av_clip_uint8`/`av_clip_uint16`. Alpha is copied.

Both `absolute`/`0` and `relative`/`1` correction-method spellings are
accepted. An all-zero range is not registered upstream and produces an
identity here. `psfile` is rejected because it reads adjustments not contained
in the graph; non-finite, out-of-range, malformed, or overlong CMYK lists are
also rejected.

## Cross-stage and platform rules

- Each stage reads the previous stage's four quantized samples, even inside a
  fused loop.
- Generated floating-point code uses `-fno-fast-math -ffp-contract=off`; the
  explicit `levels_f32_fma` operation must be implemented with an explicit
  correctly-rounded FMA or a materialized table over the format's domain rather
  than relying
  on ambient compiler contraction. The same rule is what lets `expr_f32` say
  which multiply-adds are fused: every one it does not name as `fma` is
  guaranteed to round twice.
- A kernel above eight bits loads and stores native 16-bit words. Every deep
  layout is little-endian, so the generated source `#error`s on a big-endian
  host rather than byte-swapping.
- Upstream's own bytes are not portable everywhere. `colorcontrast`,
  `vibrance`, `colortemperature`, `selectivecolor`, and some `curves` key
  points depend on whether the build target has a fused
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
