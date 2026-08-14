# Reach roadmap status

The six items below were the planned follow-up to the Week 6 MVP. This
document records what landed, what did not, and — where something did not — the
specific finding that determines how much work is left.

All six have now landed. Item 3 was the last one still half-open; its RGB half
has since landed too, together with the cross-channel float IR operation that
was blocking it. That operation now also carries `vibrance`,
`colortemperature`, and `selectivecolor`, completing the first recommended
follow-up below. Item 4 was extended with the `yuva`, remaining
subsampling-ratio, and full-range YUVJ layouts, and item 2 with 55 layouts
spanning eight through sixteen bits per component.

The framing changed twice during this work. Both changes are stated first,
because each one reorders the roadmap.

## Format coverage is a correctness gate, not a performance knob

The original plan treated pixel-format conversion as a cost: fuse an island in
RGBA, pay for a conversion in and out, and win if the island is long enough.
That is wrong. A pointwise filter produces *different bytes* in different pixel
formats, and pinned FFmpeg confirms it:

```console
$ ffmpeg -f lavfi -i "testsrc2=s=64x64:r=1" -vf "negate" -f rawvideo -pix_fmt rgba -
b93ebebdaa829ae7764b18ffbde8c8f3724a26af
$ ffmpeg -f lavfi -i "testsrc2=s=64x64:r=1" -vf "format=rgba,negate" -f rawvideo -pix_fmt rgba -
81d912efbf03e4226077bf2cccfc42041006e9fe
```

`negate` under normal negotiation runs in YUV and does not agree with the same
filter forced into RGBA. So a kernel may only replace a run that *already* works
in a format the kernel implements. Fusing a YUV or negotiation-decided island
into an RGBA kernel would silently change output.

This is why item 2 (formats) and item 4 (YUV) are not optimizations sitting
next to item 1 (islands): without them, item 1's reach cannot grow at all.
Island discovery finds the runs; format coverage decides how many of them may
actually be fused.

The same gate cuts the other way once a format *is* supported, and item 4 is
where that became load-bearing. A filter's own advertised format list is part
of the correctness argument: `lutrgb` is RGB-only, so a run written as
`format=yuv420p,negate,lutrgb,colorlevels` is not one run at all — FFmpeg
converts around `lutrgb` — and no kernel may replace it. Format coverage
decides which runs are fusible; the *filter* subset decides which runs exist.

Item 3 has now widened the second, and the gate is symmetric: `lutyuv`, `eq`,
and `hue` are YUV-only, so they are refused in an RGB run exactly as `lutrgb`
is refused in a YUV one. `colorbalance`, `colorcontrast`, `curves`, `vibrance`,
`colortemperature`, and `selectivecolor` are RGB-only and land on the same side
as `lutrgb`; `selectivecolor` narrows that side further to packed RGB.
`negate` remains the only accepted filter in both families.

## Some of upstream's own bytes are not portable

The second reframing came out of the RGB half of item 3, and it is sharper than
the first because it is a limit on what bit-exactness can even mean.

A C compiler may contract `a * b + c` into a fused multiply-add, which rounds
once where the written expression rounds twice. Clang does it by default —
`-ffp-contract=on` — but only where the target's instruction set has the
operation. FFmpeg's configure passes no `-march`, so the same pinned revision,
built by the same script, produces *different bytes* on AArch64 than on
baseline x86-64. For `colorcontrast` that is 147 bytes out of the 50,331,648
its complete `256^3` RGB domain covers: about three pixels in a million, or
roughly eighteen bytes in a 1080p frame.

Initially this compiler treated contraction as a rounding it did not decide
and refused anything that depended on it; `colorlevels` used that policy when
only eight-bit formats existed. It does not work for `colorcontrast`:
essentially every useful option set is contraction-sensitive, so refusing them
all would be the same as not implementing the filter. High depth later made the
same policy untenable for `colorlevels` too: ordinary fourteen- and sixteen-bit
option sets frequently differ at some sample.

So the IR states the contraction instead of leaving it to whichever compiler
sees the generated C. The expression operation has an explicit `fma`, the
lowering emits it at exactly the sites upstream's compiler fuses, and the
kernel is compiled with `-ffp-contract=off` so nothing else is fused behind its
back. `fma` is safe to name because IEEE-754 specifies it exactly — unlike the
`pow` and `sin` this compiler still refuses to depend on, one fused
multiply-add is the same number on every conforming target.

Which kind of host this is comes from `lavfi_cc/target.py`, a machine table
rather than a compiler probe, because the analysis-only scanner must not invoke
a compiler. A table is a claim, so a test checks it: it compiles the question,
runs it, and fails if the toolchain disagrees. `LAVFI_CC_FUSED_MULTIPLY_ADD`
overrides the table for a host that builds FFmpeg with a `-march` its baseline
does not imply.

Because the fusion decision changes which operations are in the program, it
changes the plan hash, so a kernel built for one kind of host can never be
served from a cache or a bundle to the other. Nothing extra enforces that; it
falls out of hashing the IR.

The cost is contained. Only arithmetic that can actually tell the difference
pays it: `colorlevels` and `curves` compare both evaluations across the whole
format domain and consult the host only when an output differs;
`colorcontrast` always states the host; and `colorbalance` cannot differ. See
those items below.

## 1. Automatically discover fusible islands — done

`lavfi_cc/islands.py` walks an arbitrary linear chain, tracks the working pixel
format, and reports every maximal run of filters that lower into the pixel IR.
`analyze_filtergraph(..., auto_islands=True)`, exposed as `--auto-islands`,
plans and rewrites every discovered run instead of requiring one explicit
`format=rgba` … `format=…` bracket.

What this added over the explicit-boundary path:

- The trailing `format` boundary is no longer required.
- Several islands in one graph are each fused, with their own kernel and cache
  key, in one rewrite.
- The working format survives filters that provably cannot change it. That
  `FORMAT_PRESERVING` set is geometric and timing filters only (`crop`, `hflip`,
  `pad`, `fps`, …); `scale` is deliberately excluded because it converts.

A run is only fused when its format is one the backend implements natively.
Everything else is reported and refused — see the correctness note above.

## 2. Widen pixel-format reach — done for 55 formats at 8–16 bits

Supported today: fourteen eight-bit formats, 29 deep planar formats, and four
packed 16-bit formats. In full, those are:

- packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`, and `bgr24`, plus
  `rgb48le`, `rgba64le`, `bgr48le`, and `bgra64le`;
- planar `gbrp` and `gbrap`, plus `gbrp` at 9, 10, 12, 14, and 16 bits and the
  alpha-carrying 10-, 12-, and 16-bit members upstream's accepted filters
  advertise; and
- planar `yuv444p`, `yuv422p`, and `yuv420p` at 8, 9, 10, 12, 14, and 16 bits,
  plus their alpha-carrying 8-, 10-, and 16-bit members where an accepted
  filter advertises them.

That is 20 RGB layouts and 27 YUV layouts. The exact names and per-filter
format matrix are in [`supported-filters.md`](supported-filters.md).

The packed offsets and planar plane order in `lavfi_cc/layouts.py` were taken
from the pinned binary rather than inferred from format descriptors, by
converting known component values into each family and reading the samples
back.

The reduction that keeps the layout count manageable is that operations remain
layout-independent: only the load and store ends change. Both families and
both sample widths share one addressing scheme — a channel lives in some plane
at a sample offset, consecutive groups are `step` samples apart, and a sample
is one byte through eight bits and two above. Packed is simply the one-plane
case. An alpha-less layout loads `a = 0` and stores only three components,
which is exactly what upstream does: the `colorchannelmixer` templates omit
every alpha term when `have_alpha` is unset, and the other accepted filters
cannot let an unstored alpha lane affect a stored component.

**Kernel ABI 2** replaced the single plane and stride with per-plane arrays
shaped like `AVFrame`'s `data[]` and `linesize[]`. Packed kernels use index 0
and ignore the rest, so one entry point and one slice loop now serve both
families. This is the change item 4 also needs.

Three format-dependent constraints were found and are enforced:

- A run may only be fused in a format that *every* filter in it advertises,
  otherwise FFmpeg converts in the middle of the run and one kernel is no longer
  equivalent to the filters it replaced. `colorlevels` and `colorchannelmixer`
  accept the `0rgb`/`rgb0` family that `negate` and `lutrgb` do not, so that
  family is outside the common subset. Depth narrows the overlap again:
  `eq` advertises no deep format, `hue` advertises only ten-bit deep formats,
  and `lutrgb` omits the packed deep BGR orders.
- `negate=components=…a` on an alpha-less format is a hard configuration
  failure upstream, not a silent no-op, so the compiler rejects it too.
- `negate_alpha` sets a *plane* mask that only packed RGB ignores. On a `gbrap`
  or `yuva` layout it really does negate alpha, so the same option means
  different things in different layouts. The compiler accepts it where it
  provably has no alpha effect and rejects it on every alpha-carrying planar
  layout, pointing at an explicit component mask instead. Item 4 made the
  lowering layout-aware, so honouring it is now a small change; it is still
  refused because refusing it is not wrong and an explicit mask already states
  the intent: `components=r+g+b+a` for planar RGB and
  `components=y+u+v+a` for planar YUV. What decides the legacy option upstream
  is `is_packed`, not the colour family.

### What depth changed

Every deep sample is a little-endian 16-bit word, but its legal domain is the
format's own `[0, 2^depth - 1]`. The IR kept all eight-bit spellings unchanged,
so existing plan hashes did not move; a deep plan uses the parallel
`load_rgba16`, `lut16`, `quantize_rgba16`, and `store_rgba16` operations and
records its actual depth. Generated source fails at compile time on a
big-endian host rather than reading an `le` format as a native word.

Tables are sized to the format domain rather than unconditionally to 65536
entries. A 10-bit LUT therefore has 1024 entries. The generated code clamps a
raw 16-bit sample before indexing a 9-, 10-, 12-, or 14-bit table; at eight and
sixteen bits the table already covers every value the storage type can hold and
the clamp disappears. This makes malformed input safe without claiming to
reproduce an upstream result where some accepted filters themselves read past
the end of their tables.

The arithmetic was re-derived rather than mechanically widened. The important
differences are recorded in `supported-filters.md`: `lutyuv` shifts its limited
range by `depth - 8`; deep planar `lutrgb` uses `255 << (depth - 8)` rather than
the format maximum; `colorlevels` scales option endpoints by `UINT16_MAX` for
every two-byte format; `hue` scales luma by the sample count and rotates chroma
about the depth's own midpoint; and every float expression quantizes to that
format's maximum. Contribution sums widen to `int32_t` above eight bits.

This is checked directly in `tests/test_high_depth.py`, while
`tests/test_layouts.py` runs the pinned FFmpeg oracle, interpreter, native
kernel, and patched-filter paths over every one of the 55 layouts.

## 3. More pointwise filters — done, including the follow-up RGB trio

`lutyuv`, `eq`, and `hue` are implemented and bit-exact against the pinned
oracle in every accepted YUV layout each one advertises. `colorbalance`,
`colorcontrast`, and `curves` are implemented and bit-exact in all 20 accepted
RGB layouts. `vibrance` and `colortemperature` now join them in all 20;
`selectivecolor` joins the six eight-bit and four deep packed layouts it
advertises upstream. The RGB half needed the cross-channel float IR operation
the roadmap predicted — though not for the filter the roadmap predicted; see
below — and the follow-up trio needed a small extension of that same operation.

### `lutyuv` is `lutrgb` with a different range, and the range is the whole point

`vf_lut.c` is one filter with two entry points. The difference at 8-bit depth
is which branch of `config_props` sets the per-component range: `lutrgb` gets
`0..255` for everything, while ordinary limited-range YUV under `lutyuv` gets
`16..235` for luma and `16..240` for each chroma channel. YUVJ is the
full-range exception recorded under item 4. That range reaches the expression
as `minval` and `maxval`, and through them as `clipval` and `negval`, so on luma `negval` is
`av_clip(16 + 235 - val, 16, 235)` rather than `255 - val`. `build_lut` takes
the range as a parameter and the two filters share one lowering.

Two consequences worth naming. On limited-range YUV, `lutyuv` with no options
is *not* an identity — the default expression is `clipval`, which clamps. And
the same expression under the two filters is two different tables, so they must
never collide on one cached kernel; a test pins that.

The one refusal that is stricter than upstream: `vf_lut.c` shares a single
options table across its entry points, so `lutyuv=r=…` is accepted upstream and
silently sets the *luma* expression. Both cross-family spellings are refused
here rather than honoured, because a spelling that means something other than
what it says is worth failing on.

### `eq` picks one of three code paths, and they disagree

`check_values` chooses per plane between copying the plane, a fixed-point
integer `process_c`, and a float `create_lut`. These are not refinements of one
another. With every option at its default the plane is copied; running
`process_c` with those same values would compute `brightness = -1` and subtract
one from every sample. Getting the choice wrong is an off-by-one on every pixel
of an otherwise no-op filter, so the lowering reproduces the branch exactly.

`av_clipf` takes `float` parameters, so every one of `eq`'s options is rounded
to binary32 on the way in and the clamp bounds are the binary32 neighbours of
the literals in the source. Two literals that differ as doubles but agree as
floats have to produce the same kernel, and a test pins that too.

Out-of-range values are clamped rather than refused, because upstream clamps
them; refusing would reject graphs FFmpeg accepts. What *is* refused is
anything that is not static: `eval=frame` re-evaluates per frame, and an option
naming `n`, `t`, or `r` is a different value on every frame. Only a plain
numeric literal is accepted.

### `hue` needed a new IR operation, because it mixes Cb into Cr

`hue` is the first accepted filter that is not channel-independent. Its chroma
step treats `(U, V)` as a 2D vector and rotates it, which item 4's rule — a
subsampled layout accepts only channel-independent operations — would have
refused outright.

That rule was too strong, and the correction is the structural result here.
What a subsampled layout cannot express is a mix of channels that have *no
sample in common*. Cb and Cr are sampled at exactly the same positions in every
accepted YUV format, so one loop can hold both. `PixelLayout.sampling_groups`
partitions the stored channels by sampling shift, each stage declares the
channel groups it reads across, and `validate_ir` requires each stage group to
fit inside one sampling group. A colour matrix is still refused on `yuv420p`,
for the original reason; a chroma rotation is not.

At eight bits upstream materializes the rotation as two 64 KiB tables indexed
by the `(u, v)` pair; the ten-bit path grows them with the domain. The
arithmetic behind them is a handful of int32 operations that cannot overflow
at these magnitudes — saturation is bounded at 10, so the coefficients fit in
16.16 with room to spare — so the kernel evaluates it inline instead.

Luma is a separate matter: upstream copies the plane when brightness is zero
and applies `lut_l` otherwise, and `lut_l` is the identity at zero brightness,
so one table describes both paths and the optimizer folds it away.

### Two libm dependencies, and the guard they share

`eq`'s gamma path calls `pow` and `hue`'s coefficients call `sin` and `cos`.
Neither is bit-exact across libm implementations, and neither is under this
compiler's control. Both are handled the way `colorlevels` already handles
float32 contraction: perturb the libm result by a few ULPs and refuse the
lowering if any output byte changes. A table that survives that cannot disagree
with the oracle over a rounding this compiler does not decide.

The guard needs one exemption, and finding it is the reason to state the rule
carefully. A perturbation test refuses anything sitting on a rounding tie, but
a tie is only a problem when the value reaching it is uncertain. At a zero
angle — which is what `hue=s=…` and `hue=b=…` mean, the most ordinary way to
use the filter — C fixes `cos(0)` at exactly 1 and `sin(0)` at exactly 0, and
the coefficient is then a multiple of one sixteenth for any saturation, so it
lands exactly on a tie about one time in eight. Guarding those would have
refused an eighth of all saturation-only graphs for a rounding no libm can
disagree about. The zero angle is therefore exempt, and one such tie is pinned
against the oracle as a test.

With that, the guard is cheap in practice: over 120 randomized `eq` graphs
spanning all three code paths and 45 randomized `hue` graphs it refused none,
and every one was byte-identical to the oracle.

### What is checked

`tests/test_layouts.py` runs a format-filtered matrix of 339 YUV
layout-and-chain combinations across all 27 YUV layouts through the interpreter
and pinned oracle, then checks a compiled kernel and patched FFmpeg for every
layout at `-filter_threads 4`. `tests/test_yuv_filters.py` pins the reasoning:
which upstream path each option set selects, which spellings are refused, and
the sampling-group rule including IR built by hand to prove it cannot be
dodged. The sanitizer gate includes both a deep full-width `yuva` walk and a
ten-bit subsampled one whose random 16-bit storage values exercise the table
index clamp.

Checked once by hand, not automated: `eq` against the oracle over 120 random
option sets, and `hue` over its complete `256x256` chroma domain — every
`(u, v)` pair — for 45 random option sets plus nine chosen ones covering
degrees, radians, negative saturation, and brightness. The suite's fixed chains
are the standing guard; those sweeps are what the libm exemption above was
found by.

### `expr_f32` is a transcription, not a formula

The operation in `lavfi_cc/expr.py` began as a single-assignment list of
float32 instructions — `channel`, `const`, the four arithmetic operations,
`min`, `max`, `neg`, and `fma` — plus one optional output per channel with its
own depth-specific quantizer. The follow-up trio added `abs`, `floor`,
`lrintf`, `eq`, and `gt` for `selectivecolor`; all remain straight-line scalar
operations. A channel with no output keeps the sample it arrived with, which is
what upstream does with alpha in every filter on the operation.

Everything about it is aimed at one property: that it says exactly which
roundings happen, in exactly which order. Every instruction rounds once.
Reassociating `(a + b) + c` can change stored samples, so a lowering
transcribes upstream's expression rather than simplifying it. `min` and `max`
are FFmpeg's `FFMIN` and `FFMAX` verbatim — ternaries on `>`, not
`fminf`/`fmaxf`, which return the
other operand when given two zeros of opposite sign. The quantizers use either
`lrintf` or a truncating conversion and saturate to the format's depth; both
clamp in float first, which is the same mapping for every input C defines and
is defined for the ones it is not. And `fma` is there for the reason stated at
the top of this document.

The interpreter and the C generator sit next to each other in that file,
because the only thing that makes them correct is that they agree instruction
for instruction.

The sampling-group rule `hue` introduced is what tells the operation where it
may run, exactly as predicted: it reads across all four channels, so it is
admissible on every RGB layout and on `yuv444p` and refused on `yuv422p` and
`yuv420p`. `validate_ir` enforces it for IR built by hand too.

### `colorbalance` is the filter the extension was for

Its per-pixel lightness term couples all three channels, which is what neither
a per-channel `lut8` nor a `matrix4x4` can express:

```c
const float l = (FFMAX3(r, g, b) + FFMIN3(r, g, b));
r = get_component(r, l, shadows, midtones, highlights);
```

`get_component` contains four multiply-adds, and every one of them scales by
`4.f`. Scaling by a power of two is exact, so fusing the multiply into the add
rounds the same value once either way, and the lowering needs to know nothing
about the host. That is not a comment: `expr.multiply_is_exact` is asked, with
the operand bound the expression actually reaches, and the lowering fails if it
ever stops holding. A contracted and a non-contracted build agreed on all
50,331,648 bytes of the complete RGB domain, three option sets each.

`pl=1` is refused. `preservel` scales its multiply-adds by a computed
saturation rather than by a power of two, so accepting it would make
`colorbalance` target-dependent for one option, which is not worth it when
`pl=0` is the default and the whole rest of the filter is portable.

### `colorcontrast` is where the contraction had to be stated

Its `PROCESS` macro is fifteen linear operations, a lightness ratio, and three
`lerpf`s, with eighteen multiply-adds that Clang contracts and none of which is
exact. The lowering emits those eighteen as `fma` when the host fuses and as
separate operations when it does not.

Finding the eighteen sites is the part that could have been wrong, so it was
not reasoned about. A model with explicit `fmaf` calls at the predicted sites
was compiled at `-ffp-contract=off` and compared against upstream's expression
compiled at `-ffp-contract=on`, over the complete `256^3` domain, for three
option sets: zero differing bytes out of 50,331,648 each. The eighteen match
the eighteen fused instructions in the pinned oracle's `colorcontrast_slice8`.

One more thing had to be reproduced rather than inferred. Upstream's slice loop
is `y < slice_end && sum > FLT_EPSILON`, so a weight sum at or below
`FLT_EPSILON` leaves the frame untouched — and the three weights default to
zero, which means `colorcontrast` with no weights is an identity no matter what
its contrast options say. The lowering emits a program that stores nothing, and
the identity pass removes the stage entirely.

### `curves` needed no IR extension at all

The roadmap listed `curves` with the other two. That was wrong, and it is worth
saying why, because the mistake was in the wrong place.

`curves`' slice functions are `dst[x + r] = graph[R][src[x + r]]` and nothing
else: one `1 << depth`-entry table per colour channel, alpha copied. It is
channel-independent, and it lowers to the same depth-sized LUT `lutrgb` lowers
to.
Everything difficult about it happens once, in `config_input`, where key points
become a curve — a natural cubic spline or PCHIP, both in `double`, with a
tridiagonal solve, edge-case derivatives, and a master curve composed on top of
each component afterwards. `lavfi_cc/curves.py` is that, and it is the largest
part of this item by line count while contributing no IR at all.

The cross-channel work was still the thing blocking `curves`, but only in the
sense that it was blocking the item. What was blocking `curves` itself was
nobody having read `vf_curves.c`'s slice functions.

Its `double` arithmetic contracts too, and here the two evaluations are built
and compared for all 256 entries, which is cheap. They almost always agree, and
then the table — and so the plan hash — is the same on every host, and the
target is never consulted. When they disagree the host decides, for the same
reason `colorcontrast` does.

They disagree more often than a `double` rounding gap suggests, because
ordinary key points land exactly on byte boundaries. `curves=r='0/0.05 1/1'` is
the straight line `y = 0.05 + 0.95x`, which at input 155 is exactly `160/255`;
one evaluation order lands just below and the other just above, and the
truncating `CLIP` turns that into 159 or 160. The pinned oracle on this host
produces 159, which is the fused answer. Nothing about that curve is unusual,
which is the point.

### The follow-up trio stays on `expr_f32`, with one extension

`vibrance` normalizes RGB, derives saturation and a configurable luma from all
three channels, then applies a separate balance to each channel. The pinned
AArch64 build contracts eleven written multiply-add sites: two in luma, six in
the three saturation coefficients, and three final `lerpf`s. The lowering
states all of them; a non-contracting target keeps the source's separate,
left-associative evaluation. Zero effective intensity is an identity on a
fused target, but separate `(input-luma)+luma` can lose a low bit, so the
default remains an expression on a non-contracting host. Alpha is copied.

`colortemperature` has two stages upstream. `kelvin2rgb` runs once per frame
and obtains three binary32 multipliers through `logf` or `powf`; the lowering
runs that calculation once with the host C library, records the resulting
constants in the plan, and emits no per-pixel libm call. The pixel stage mixes
the scaled RGB, optionally restores its input lightness, and has six contracted
`lerpf` sites. `mix=0` is an identity; otherwise the plan states the host's
contraction. Alpha is copied.

`selectivecolor` is narrower in format even though it is not narrower in
arithmetic: upstream advertises only packed RGB, so it is accepted on the six
eight-bit and four packed 16-bit layouts and refused on every `gbrp` layout.
Its nine color ranges classify one pixel by integer minimum, middle, maximum,
and midpoint tests. Each active range computes three CMYK adjustments, rounds
each with `lrintf`, and only then adds the integers from different ranges. One
final float output cannot recover those intermediate boundaries.

That is the extension to `expr_f32`: exact `eq`/`gt` predicates, `abs` and
`floor` for the integer range scales, and an intermediate `lrintf`. Predicate
results and rounded range terms remain exactly representable as binary32 at
both supported widths, so the expression is still a straight-line
single-assignment program. The one multiply-add in each component adjustment
is stated as fused or separate. Both correction methods and one-to-four-value
CMYK option strings are accepted; `psfile` is refused because its adjustments
are not in the graph. Alpha is copied.

### What the RGB filters are checked against

In the suite: `tests/test_layouts.py` now runs a format-filtered matrix of 346
RGB layout-and-chain combinations across all 20 RGB layouts through the
interpreter and pinned oracle, then checks a compiled kernel and patched FFmpeg
for every layout at `-filter_threads 4`. `tests/test_rgb_filters.py` pins the reasoning — the
expression IR's validation and evaluation rules, the sampling-group refusal
including IR built by hand, which option spellings are refused and why, that
`colorbalance` has one plan hash on both kinds of host and `colorcontrast` two,
that the new active float filters likewise separate contracting and
non-contracting hosts, and the toolchain check that turns the machine table
into a claim under test. The ASan/UBSan inline-expression cases now include all
three additions, including selective color's predicates and intermediate
rounding.

Checked once by hand, not automated:

- `colorbalance` and `colorcontrast` against the oracle over their **complete**
  `256^3` RGB domain — every pixel — for five option sets: 0 differing bytes
  out of 50,331,648 each. This is the check that the eighteen fusion sites and
  the whole transcription are right, and it leaves no input untested.
- 120 randomized option sets (40 per filter, `curves` over both interpolators
  with random key points) on a 1024×1024 random frame: no mismatches.
- 108 randomized follow-up option sets (36 per filter) across packed eight-bit,
  planar 10/16-bit, and packed 16-bit RGB frames: no mismatches. Chosen native
  chains containing all three additions also matched the oracle at both sample
  widths.
- Generated C for all 85 layout-and-chain combinations that existed before this
  item is byte-identical, so no cached or bundled kernel was invalidated.

### The oracle had to be rebuilt with `--enable-gpl`

`vf_eq.c` is GPL, and the pinned oracle was configured without `--enable-gpl`,
so it had no `eq` filter at all — the filter could be implemented but not
checked, which for this compiler is the same as not implementing it. Both build
scripts now pass `--enable-gpl` at the same pinned revision `n8.1.2`. The
binaries in `.build/` are GPL rather than LGPL as a result; they are local
artifacts and are not distributed. `lutyuv` and `hue` are LGPL and were present
all along.

## 4. Avoid boundary conversion — done for 35 planar YUV formats

The roadmap offered two routes: fuse the format conversion into the island
boundary, or introduce a YUV IR. The first should not be attempted. Fusing the
conversion means reproducing swscale byte-exactly, and swscale selects among
many code paths; matching all of them is a far larger commitment than the
filters themselves.

The second route is the one taken, and it is better described as *avoiding*
conversion rather than fusing it: planar YUV is now supported natively, so a
YUV-native island fuses in place with no conversion at either boundary.
Bit-exactness is against the filters, which is tractable, instead of against
swscale, which is not.

### Subsampling made the pixel loop a plane loop

The pixel IR still describes four logical channels, and it is still
layout-independent. What changed is that a layout can now say each channel is
stored at a *fraction* of the frame's resolution, and `width`/`height` mean
plane 0 only. Every other plane's dimensions come from `AV_CEIL_RSHIFT`, which
is how the rest of libavfilter sizes a chroma plane, and which rounds up: a
`5x3` `yuv420p` frame has `3x2` chroma planes, not `2x1`.

That has a consequence beyond addressing. With subsampling there is no longer
one loop iteration per pixel, so there is no single pixel whose four channels
an operation could mix. Item 4 stated that rule as *a subsampled layout accepts
only channel-independent operations*, which was too strong; item 3's `hue`
corrected it to what it should have been. The real constraint is that an
operation may only read across channels that share a sample grid, and `hue`'s
chroma rotation does. See item 3 for the sampling-group formulation that
replaced it. `colorchannelmixer` is refused on `yuv422p` and `yuv420p` either
way, and `validate_ir` still enforces it for the interpreter, the optimizer,
and code generation alike.

Both the interpreter and the code generator therefore have two shapes: a
whole-pixel walk for a layout whose planes share the frame's resolution — which
is every RGB layout, plus `yuv444p` — and a walk over sampling groups at each
group's own resolution for `yuv422p` and `yuv420p`. A group is one plane
whenever every stage is channel-independent, which is every pipeline that does
not contain `hue`; generated C for all 55 layout-and-chain combinations that
existed before item 3 is byte-identical, so no cached or bundled kernel was
invalidated by either item.

### The layout reached the lowering, and the option names moved with it

`negate` was the only accepted filter that advertised YUV at all when item 4
landed; `lutrgb`, `colorlevels`, and `colorchannelmixer` are RGB-only upstream,
so a YUV run containing one of them is refused rather than fused. That refusal
is not a detail: the corpus graph `format=yuv420p,negate,lutrgb=…,colorlevels=…`
used to be reported as a two-pass island waiting on YUV support, and it never
was one — FFmpeg converts around `lutrgb`, so those two passes were never
removable. Supporting YUV is what made that visible. Item 3 has since added
`lutyuv`, `eq`, and `hue` on the YUV side, so a YUV island is no longer built
out of `negate`s alone.

`negate`'s component mask is named per family, and upstream fails to configure
a graph that names a component the format lacks. `components=r+g+b` is a hard
error on `yuv420p` and `components=y+u+v` is one on `rgba`. Expressing that
required the lowering to take the layout, which is the signature change item 2
had deferred. `negate_alpha` needed no new rule: it selects plane 3, and the
accepted YUV formats have three planes, so it is inert there exactly as it is on
`gbrp`.

### Slice threading had to learn the chroma grid

`vf_fused.c` previously offset every plane by the same starting row, which is
correct only when no plane is subsampled. It now cuts slices in units of
`1 << log2_chroma_h` plane-0 rows, so each chroma row belongs to exactly one
job and the per-plane row counts tile exactly with no overlap. Without that,
two jobs would write the same chroma sample.

### What this is checked against

In the suite: byte-exact against the pinned oracle through the interpreter, the
compiled kernel, and patched FFmpeg, for every accepted layout.
`tests/test_layouts.py` runs at `17x5` and `21x7`, odd in both dimensions so the
`AV_CEIL_RSHIFT` rounding is exercised, and the end-to-end case passes
`-filter_threads 4` so the slice alignment is too. The sanitizer gate covers
eight-bit and deep packed, planar, subsampled, and full-resolution-alpha kernel
shapes, with each plane's geometry passed to the harness rather than duplicated
in C.

Checked once by hand, not automated: a sweep over `1x1`, `2x3`, `3x2`, `17x9`,
`64x33`, `65x64`, and `127x71` at one, three, and eight filter threads, which
was byte-identical to the oracle in all three YUV formats. The suite's fixed
sizes are the standing guard; that sweep was how the slice alignment was
convinced to be right at job counts the suite does not reach.

Generated C for the pre-existing layouts was byte-identical to before item 4
across all 40 layout-and-chain combinations, item 3 preserved that property
across all 55 that existed by then, and `yuva` preserved it across all 125, so
no cached or bundled kernel was invalidated by any of them.

### `yuva*` was the prediction the sampling-group rule was waiting for

`yuva444p`, `yuva422p`, and `yuva420p` are now supported too, and they are the
first layouts that subsample *and* carry alpha. Every earlier layout had alpha
only where nothing was subsampled, so "does this layout subsample?" and "may an
operation read across channels?" had never come apart.

Only chroma shrinks. Alpha keeps the frame's resolution — upstream's `negate`
writes `height[0] = height[3] = inlink->h` and applies `AV_CEIL_RSHIFT` to
planes 1 and 2 only — so `yuva420p` partitions into `(luma, alpha)` and
`(Cb, Cr)`. The roadmap predicted three sampling groups; there are two, because
alpha is co-sited with luma rather than being a grid of its own. That is the
one thing about this item that was called wrong in advance, and it makes the
layout *more* expressive than expected rather than less: an operation reading
luma together with alpha is admissible on `yuva420p`, for exactly the reason
`hue`'s rotation is. No accepted filter writes one, so the rule is pinned by IR
built by hand.

What did need care is that the two groups are not adjacent planes. A kernel
therefore contains loops at two different resolutions, with plane 3 walked at
plane 0's row count; a plane-3 pointer advanced by a chroma plane's row count
would run off the end of the frame. `vf_fused.c`'s slice loop needed no change
for that — it already offset planes 1 and 2 by the chroma grid and 0 and 3 by
the full one, which was written for the alpha-less case and turns out to be the
general rule. The filter's only change is three more format identifiers.

The filter subset needed no widening either: every filter that advertises a
planar YUV format at this depth advertises its alpha-carrying twin, so `negate`,
`lutyuv`, `eq`, and `hue` all reach `yuva` unchanged. What changed is what
alpha *means* to them, and each one had to be read rather than assumed. `eq`
copies plane 3 outright (`if (i == 3 || !adjust)`), `hue` copies it, and
`lutyuv` applies a real table to it — built from component A's `0..255` range
rather than luma's `16..235`, so the same `negval` expression is two different
tables in one filter invocation. All three were already lowered that way; what
was new is that the alpha channel is now stored, so being wrong would show.

Two `negate` option rules move with the alpha plane, and both are ones `gbrap`
already had:

- `negate=components=…a` is a hard configuration failure on `yuv420p` and
  accepted on `yuva420p`. That is the entire difference between the two
  formats as far as the option is concerned.
- `negate_alpha=1` really does negate alpha here, exactly as on `gbrap`, since
  what decides it upstream is `is_packed` rather than the colour family. It is
  refused, and the refusal now names a spelling in the layout's own family:
  `components=y+u+v+a`, because suggesting `components=r+g+b+a` on a YUV format
  would be suggesting another configuration failure.

#### What the `yuva` half is checked against

The plane order was read out of the pinned binary rather than from a
descriptor, the same way every other layout was: one `negate` per component
over a frame of four distinct plane values moves plane 0 for `y`, 1 for `u`, 2
for `v`, and 3 for `a`, and plane 3 is `width * height` bytes rather than a
chroma plane's size.

In the suite: `tests/test_layouts.py` runs twenty-two chains across the three
new layouts — the fifteen YUV chains plus seven that reach alpha, which the
alpha-less trio cannot express at all — through the interpreter, the compiled
kernel, and patched FFmpeg at `-filter_threads 4`. `tests/test_yuv_filters.py`
pins the reasoning: the `(luma, alpha)` partition, that a luma-and-alpha
expression is admissible where a colour matrix still is not, `lutyuv`'s alpha
range, and both alpha option refusals with the family-correct spelling. The
ASan/UBSan gate gained a fifth case for the two-resolution kernel shape.

Checked once by hand, not automated:

- The size-and-threads sweep above, repeated for the three `yuva` formats: 63
  runs over `1x1`, `2x3`, `3x2`, `17x9`, `64x33`, `65x64`, and `127x71` at one,
  three, and eight filter threads, all byte-identical to the oracle through
  patched FFmpeg. This is what convinces the full-resolution alpha plane to
  tile correctly against chroma-aligned slice boundaries.
- 75 randomized alpha-touching option sets (25 per layout, over `negate` masks
  including `a`, `lutyuv` alpha expressions, `eq`, and `hue`) on a `129x67`
  frame: no mismatches and no refusals.

### The remaining YUV table entries are done

`yuv411p`, `yuv410p`, `yuv440p`, and `yuv440p10le` now use the same generalized
plane walk with chroma shifts `(2,0)`, `(2,2)`, `(0,1)`, and `(0,1)`
respectively. The two-bit vertical shift in 4:1:0 also exercises the existing
slice rule at a four-row chroma grid rather than the two-row grid of 4:2:0.

The supported deprecated full-range aliases are `yuvj444p`, `yuvj422p`,
`yuvj420p`, and `yuvj440p`. Their geometry matches the ordinary formats, but
their `lutyuv` semantics do not: in pinned `vf_lut.c` they miss the
limited-range YUV switch arm and use `0..255` for luma and both chroma
components. `PixelLayout.full_range` records that distinction, so `clipval` is
an identity and `negval` is `255-val` on YUVJ instead of using the
`16..235`/`16..240` studio ranges. `yuvj411p` is not included because none of
the accepted filters advertises it in the pinned revision.

The filter-format table follows the upstream lists rather than treating these
layouts as one block: `negate` and `lutyuv` accept all eight additions; `eq`
accepts only `yuv411p` and `yuv410p`; and `hue` accepts the four non-YUVJ
ratios, including `yuv440p10le`. Each layout has its own kernel ABI identifier
and patched-filter mapping. The layout differential matrix now sends every
advertised chain through the interpreter and native kernel against pinned
FFmpeg, while focused tests pin the full-range LUTs, filter lists, and all four
non-JPEG layout geometries.

## 5. Analysis-only scanner — done

`lavfi-cc scan` consumes arbitrary filtergraphs — several chains, link labels,
whitespace, hardware filters, options this compiler cannot parse — and never
compiles or runs anything. A lenient parser records per-filter option problems
instead of failing the graph, so a scan still explains what blocked an island.

The corpus in `tests/corpus/filtergraphs.txt` grew from 40 graphs to 45 in item
4, to 58 in item 3's YUV half, to 69 in its RGB half, to 77 with `yuva`, and to
89 with high-depth formats. The last twelve are nine grading chains over
10-, 12-, and 16-bit YUV/RGB plus three same-family format refusals whose answer
depends on depth. The `yuva` addition was five grading chains on video that
carries an alpha plane and three graphs where the alpha plane itself decides
the refusal. The earlier YUV additions are
seven grading chains of the shape people actually write — `eq` and `hue`
together, sometimes with `lutyuv` or a `crop` between them — four graphs where
an accepted filter is pinned to a format it does not advertise, and three where
a YUV option is inside the subset but still rejected. The RGB additions are
eight grading chains built from `curves`, `colorbalance`, and `colorcontrast`
and three more rejected-option graphs, one per new refusal.

The three follow-up graphs for `colortemperature`, `vibrance`, and
`selectivecolor` have now moved out of the "outside the pointwise subset"
section and into RGB grading. The remaining examples there are genuinely
non-pointwise filters rather than filters waiting on an existing operation.

```console
$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt
  graphs scanned:            89
  islands found:             87
  islands fusible today:     73
  frame passes eliminated:   79
  frame passes blocked:      10
  blocked passes by working format:
    negotiated  10 passes across 14 islands

$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt --entry-format yuv420p
  frame passes eliminated:   82
  frame passes blocked:      4
```

Blockers are ranked by the frame passes they withhold rather than by how often
they appear, which is what makes the output actionable.

Item 2's format work moved this corpus from 17 eliminated passes to 19 with
`rgb24`, then to 21 with planar `gbrp`, dropping blocked passes from 12 to 8.

Item 4's effect is best read on the corpus as it stood *before* its five new
graphs, which separates the change from the sample it is measured on:

| the same 40 graphs | eliminated | blocked |
|---|---:|---:|
| before item 4 | 21 | 8 |
| after item 4 | 21 | 6 |
| after item 4, `--entry-format yuv420p` | 23 | 2 |

The first row of movement is a correction, not a gain. Two of those eight
blocked passes were the `format=yuv420p,negate,lutrgb,colorlevels` island, and
it was never fusible at any point: `lutrgb` is RGB-only, so FFmpeg converts
around it and the run is not contiguous in one format. Supporting YUV is what
let the scanner see that, and the 2 passes it was crediting were never
removable.

The gain appeared under `--entry-format yuv420p`, which is what a caller can
prove for decoded video: `fps=30,negate,negate,negate` is a genuine YUV island
that a kernel now replaces.

Item 3's effect is measured the same way, on the 58-graph corpus with and
without the three new filters, so the sample is held fixed:

| the same 58 graphs | eliminated | blocked |
|---|---:|---:|
| before `lutyuv`, `eq`, `hue` | 27 | 8 |
| after | 36 | 9 |
| after, `--entry-format yuv420p` | 39 | 4 |

Nine more frame passes, and this time it is a gain rather than a correction:
these are runs that exist in one format and that a kernel can now replace.
Measured on the *unchanged* 45-graph corpus the movement is only 27 to 28,
which is the honest reading of how little YUV grading that corpus contained —
one graph, `format=yuv444p,lutyuv=…,eq=…`, which item 4 could only report.

The one blocked pass that appeared is real rather than a regression:
`fps=30,eq=…,hue=…` is a newly visible island that negotiation still decides
the format of, so it needs an entry format a caller can prove. It was not
counted before because `eq` and `hue` were not in the subset and there was no
island there to block.

The RGB half is measured the same way, on the 69-graph corpus with and without
its three filters:

| the same 69 graphs | eliminated | blocked |
|---|---:|---:|
| before `colorbalance`, `colorcontrast`, `curves` | 36 | 10 |
| after | 47 | 11 |
| after, `--entry-format yuv420p` | 50 | 5 |

Eleven more frame passes, and again one newly visible blocked island:
`curves=…,colorbalance=…,negate` with no `format` filter is a genuine run whose
format negotiation still decides.

Measured on the *unchanged* 58-graph corpus the movement is 36 to 36 — no gain
at all, even though three more islands are found and fused. That corpus
contained these filters only as isolated single-filter examples, and replacing
one filter with one kernel removes no frame pass. It is a useful reminder of
what the scanner's headline number measures: not how many filters are
supported, but how many frame passes a run of them removes.

`yuva` is measured the same way again, on the 77-graph corpus with and without
the three new layouts in the backend:

| the same 77 graphs | eliminated | blocked |
|---|---:|---:|
| before `yuva444p`, `yuva422p`, `yuva420p` | 47 | 20 |
| after | 54 | 11 |
| after, `--entry-format yuv420p` | 57 | 5 |

Seven more frame passes eliminated, but nine fewer blocked, and the gap is the
interesting part. Two of those nine were never removable, and neither was
visible as such until the scanner could see the format:

- `format=yuva420p,curves=…,negate` was reported as one two-filter island.
  `curves` is RGB-only, so FFmpeg converts around it and the run was never
  contiguous — the same correction `format=yuv420p,negate,lutrgb,colorlevels`
  produced in item 4, arriving for the same reason.
- `format=yuva420p,negate=negate_alpha=1,negate` is a genuine two-filter run in
  one format, and this compiler refuses it rather than honour a plane mask that
  means something different here than in packed RGB. That pass is withheld by a
  decision recorded above, not by FFmpeg, and it is the one place where the
  numbers move because of a refusal rather than a fact.

Measured on the *unchanged* 69-graph corpus the movement is 47 to 47: no gain,
because that corpus contained no alpha-carrying YUV graph at all. As with the
RGB half, the headline number measures runs removed, not formats supported.

High depth adds twelve graphs to that 77-graph corpus. The nine grading chains
account for all nineteen newly eliminated passes, taking the total from 54 to
73; the three refusal graphs make the scanner state that `eq` is eight-bit
only, `lutyuv` and `hue` do not overlap on deep `yuva`, and `lutrgb` omits the
packed deep BGR orders. On the unchanged 77 graphs, widening the backend alone
does not move the headline count because none of those graphs names a deep
format. That is the same corpus lesson as the RGB and `yuva` additions: reach
only becomes measurable after the sample contains it.

The follow-up RGB trio is measured on that same unchanged 89-graph corpus; its
three examples were already present as refusals:

| the same 89 graphs | eliminated | blocked |
|---|---:|---:|
| before `vibrance`, `colortemperature`, `selectivecolor` | 73 | 12 |
| after | 77 | 12 |
| after, `--entry-format yuv420p` | 80 | 6 |

Four more frame passes disappear. The number of islands falls from 88 to 87
because a previously unsupported filter no longer splits two supported
neighbours; that is a merge, not lost reach. Blocked passes do not move because
all three grading graphs pin `rgba` explicitly.

Item 4 removed the format barrier and named the filter subset as the next
binding constraint. Item 3 widened the subset on both sides, then `yuva` and
high depth widened format reach again. The last YUV table entries remove the
only format blocker named by this corpus: its `yuv410p` chain now eliminates
two more passes, leaving all ten format-attributed blocked passes under
negotiation rather than a known unsupported layout. What binds now is the
filter subset and the caller knowledge needed to pin negotiation-decided runs.

## 6. Build-time integration — done for the compiler, unchanged for the patch

**Runtime Clang is now optional.** `lavfi-cc bundle` compiles the kernels a
corpus needs once, at build time, into a directory indexed by plan hash:

```sh
./lavfi-cc bundle --file graphs.txt --auto-islands --output kernels/
./lavfi-cc run --bundle kernels/ --require-bundle -- ffmpeg-arguments…
```

`run` consults the bundle before the cache and validates a hit by checksum,
ABI, layout, and plan hash, so a stale or tampered bundle is rejected rather
than trusted. `--require-bundle` refuses to invoke a compiler at all. This is
covered by a test that runs fusion with `PATH=/var/empty` and a nonexistent
`LAVFI_CC_CLANG` and still produces byte-identical output.

`--emit-only` writes the generated C and the index without compiling, so a
project's own build system can compile the kernels with its own toolchain. That
path needs no Clang at any point.

**The patched FFmpeg is still required**, and this is a structural limit rather
than an unfinished task: FFmpeg has no stable plugin ABI, so an out-of-tree
filter cannot be loaded by a stock binary. The realistic paths are upstreaming
`vf_fused.c` or shipping a patched build. The filter did become more
upstream-shaped here: it now negotiates formats dynamically from the kernel it
loaded via `FILTER_QUERY_FUNC2` instead of hardcoding a single format.

`scripts/build-ffmpeg-week5.sh` also now refreshes `vf_fused.c` on every build.
It previously only copied the filter when creating the worktree, so edits were
silently ignored on rebuild.

## Recommended order from here

1. **Let a real corpus select the next pointwise filter.** The roadmap's flat
   format backlog is now empty. The scanner's remaining format-attributed
   blockers are negotiation-decided rather than known layouts the backend lacks, while
   its common unsupported filters (`crop`, `scale`, and `fps`) are structural
   or frame-global rather than missing pixel operations.

Six entries left this list by being done. Layout-aware lowering landed as
part of item 4, because `negate`'s component mask could not be expressed in YUV
without it. `lutyuv`, `eq`, and `hue` landed as item 3's YUV half. The
cross-channel float operation landed as `expr_f32`, with `colorbalance`,
`colorcontrast`, and `curves` on top of it — and `curves`, which was expected
to need it, turned out not to. `yuva*` landed as part of item 4, and was the
one entry whose predicted shape was wrong: two sampling groups rather than
three, because alpha is co-sited with luma. High-depth support then widened the
same load, store, table, quantizer, expression, interpreter, ABI, and FFmpeg
paths across the 33 accepted deep layouts without changing the eight-bit IR.
The former first recommendation has now landed too: `vibrance` and
`colortemperature` use the existing arithmetic directly, while
`selectivecolor` added exact predicates and intermediate rounding to
`expr_f32` so its nine independently rounded range adjustments remain
bit-exact. The former YUV-table recommendation has now landed as well: eight
more layouts take the total to 55, with YUVJ's full-range LUT behavior recorded
rather than inferred from its otherwise identical geometry.
