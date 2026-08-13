# Reach roadmap status

The six items below were the planned follow-up to the Week 6 MVP. This
document records what landed, what did not, and — where something did not — the
specific finding that determines how much work is left.

All six have now landed at least in part. Item 3 is the most recent and is the
only one still half-open: its YUV filters are done and its RGB filters are not,
for a reason recorded under that item.

The framing changed once during this work, and it is worth stating first
because it reorders the whole roadmap.

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
is refused in a YUV one. `negate` remains the only accepted filter in both
families.

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

## 2. Widen pixel-format reach — done for 8-bit RGB and planar YUV, deferred above that

Supported today: packed `rgba`, `bgra`, `argb`, `abgr`, `rgb24`, `bgr24`,
planar RGB `gbrp`, `gbrap`, and planar YUV `yuv444p`, `yuv422p`, `yuv420p`
(item 4).

The byte offsets and the planar plane order in `lavfi_cc/layouts.py` were taken
from the pinned binary rather than from format descriptors, by converting one
known `0x11223344` pixel into each format and reading the bytes back.

The reduction that makes this cheap: the operations are layout-independent, so
only the load and store ends change. Both families share one addressing scheme
— a channel lives in some plane at some offset, with samples `step` bytes apart
— so packed is simply the one-plane case. An alpha-less layout loads `a = 0`
and stores only three components, which is exactly what upstream does: the
`colorchannelmixer` templates omit every alpha term when `have_alpha` is unset,
and the other accepted filters treat channels independently, so an alpha lane
that is never stored cannot affect the stored ones.

**Kernel ABI 2** replaced the single plane and stride with per-plane arrays
shaped like `AVFrame`'s `data[]` and `linesize[]`. Packed kernels use index 0
and ignore the rest, so one entry point and one slice loop now serve both
families. This is the change item 4 also needs.

Three format-dependent constraints were found and are enforced:

- A run may only be fused in a format that *every* filter in it advertises,
  otherwise FFmpeg converts in the middle of the run and one kernel is no longer
  equivalent to the filters it replaced. `colorlevels` and `colorchannelmixer`
  accept the `0rgb`/`rgb0` family that `negate` and `lutrgb` do not, so that
  family is outside the common subset.
- `negate=components=…a` on an alpha-less format is a hard configuration
  failure upstream, not a silent no-op, so the compiler rejects it too.
- `negate_alpha` sets a *plane* mask that only packed RGB ignores. On `gbrap`
  it really does negate alpha, so the same option means different things in
  different layouts. The compiler accepts it where it provably has no alpha
  effect and rejects it on `gbrap`, pointing at `components=r+g+b+a` instead.
  Item 4 made the lowering layout-aware, so honouring it on `gbrap` is now a
  small change; it is still refused because refusing it is not wrong and
  `components=r+g+b+a` already states the intent.

**Deferred:** the 9–16-bit formats, which need 65536-entry tables and a
re-derivation of every quantization rule at that depth.

## 3. More pointwise filters — the YUV half landed, the RGB half is unchanged

`lutyuv`, `eq`, and `hue` are implemented and bit-exact against the pinned
oracle in all three accepted YUV layouts. The RGB half of the item —
`curves`, `colorbalance`, and `colorcontrast` — is untouched, because it still
needs an IR extension rather than a new table.

### `lutyuv` is `lutrgb` with a different range, and the range is the whole point

`vf_lut.c` is one filter with two entry points. The difference at 8-bit depth
is which branch of `config_props` sets the per-component range: `lutrgb` gets
`0..255` for everything, `lutyuv` gets `16..235` for luma and `16..240` for
each chroma channel. That range reaches the expression as `minval` and
`maxval`, and through them as `clipval` and `negval`, so on luma `negval` is
`av_clip(16 + 235 - val, 16, 235)` rather than `255 - val`. `build_lut` takes
the range as a parameter and the two filters share one lowering.

Two consequences worth naming. `lutyuv` with no options is *not* an identity —
the default expression is `clipval`, which clamps into the limited range. And
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

Upstream materializes the rotation as two 64 KiB tables indexed by the `(u, v)`
pair. The arithmetic behind them is a handful of int32 operations that cannot
overflow at these magnitudes — saturation is bounded at 10, so the coefficients
fit in 16.16 with room to spare — so the kernel evaluates it inline instead:
same bytes, 128 KiB less generated C.

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

`tests/test_layouts.py` runs sixteen YUV chains across all three layouts
through the interpreter, the compiled kernel, and patched FFmpeg at
`-filter_threads 4`. `tests/test_yuv_filters.py` pins the reasoning: which
upstream path each option set selects, which spellings are refused, and the
sampling-group rule including IR built by hand to prove it cannot be dodged.
The ASan/UBSan gate gained a third case for the two-planes-in-one-loop shape.

Checked once by hand, not automated: `eq` against the oracle over 120 random
option sets, and `hue` over its complete `256x256` chroma domain — every
`(u, v)` pair — for 45 random option sets plus nine chosen ones covering
degrees, radians, negative saturation, and brightness. The suite's fixed chains
are the standing guard; those sweeps are what the libm exemption above was
found by.

### The RGB half is unchanged, and `colorbalance` is still the clearest example

`curves`, `colorbalance`, and `colorcontrast` are RGB-native and **not**
channel-independent. `hue`'s chroma rotation does not help them: it mixes two
channels that a table could not, but it is one fixed integer operation, not the
general float expression these need. `colorbalance`'s per-pixel lightness term
couples all three channels:

```c
const float l = (FFMAX3(r, g, b) + FFMIN3(r, g, b));
r = get_component(r, l, shadows, midtones, highlights);
```

Neither a per-channel `lut8` nor a `matrix4x4` can express that. It is still
pixel-local and therefore fusible in principle, but it needs a new IR operation
evaluating a fixed float32 expression across all four channels, with upstream's
exact operation order, `av_clipf` clamping, and `lrintf` ties-to-even rounding
reproduced. That is a genuine IR extension, not a new table.

The sampling-group rule `hue` introduced does tell that extension where it may
run: such an operation reads across all four channels, so it fits `yuv444p` and
every RGB layout and is refused on `yuv422p` and `yuv420p` — which is the
guard that already applies to `colorchannelmixer`.

Adding a filter whose semantics are only approximately right would be worse
than not adding it, because bit-exactness against the pinned oracle is the
property this compiler exists to preserve.

### The oracle had to be rebuilt with `--enable-gpl`

`vf_eq.c` is GPL, and the pinned oracle was configured without `--enable-gpl`,
so it had no `eq` filter at all — the filter could be implemented but not
checked, which for this compiler is the same as not implementing it. Both build
scripts now pass `--enable-gpl` at the same pinned revision `n8.1.2`. The
binaries in `.build/` are GPL rather than LGPL as a result; they are local
artifacts and are not distributed. `lutyuv` and `hue` are LGPL and were present
all along.

## 4. Avoid boundary conversion — done for `yuv444p`, `yuv422p`, `yuv420p`

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

In the suite: byte-exact against the pinned `.build/ffmpeg-macos` oracle through
the interpreter, the compiled kernel, and patched FFmpeg, for every accepted
layout. `tests/test_layouts.py` runs at `17x5` and `21x7`, odd in both
dimensions so the `AV_CEIL_RSHIFT` rounding is exercised, and the end-to-end
case passes `-filter_threads 4` so the slice alignment is too. The sanitizer
gate builds and runs a `yuv420p` kernel alongside the packed one, with each
plane's geometry passed to the harness rather than duplicated in C.

Checked once by hand, not automated: a sweep over `1x1`, `2x3`, `3x2`, `17x9`,
`64x33`, `65x64`, and `127x71` at one, three, and eight filter threads, which
was byte-identical to the oracle in all three YUV formats. The suite's fixed
sizes are the standing guard; that sweep was how the slice alignment was
convinced to be right at job counts the suite does not reach.

Generated C for the pre-existing layouts was byte-identical to before item 4
across all 40 layout-and-chain combinations, and item 3 preserved that property
across all 55 that existed by then, so no cached or bundled kernel was
invalidated by either.

**Deferred:** `yuva*` (alpha at full resolution alongside subsampled chroma),
the `yuvj` full-range family, and the remaining ratios `yuv411p`, `yuv410p`,
and `yuv440p`. Each is a table entry rather than a design question, but none of
them appears in the corpus.

## 5. Analysis-only scanner — done

`lavfi-cc scan` consumes arbitrary filtergraphs — several chains, link labels,
whitespace, hardware filters, options this compiler cannot parse — and never
compiles or runs anything. A lenient parser records per-filter option problems
instead of failing the graph, so a scan still explains what blocked an island.

The corpus in `tests/corpus/filtergraphs.txt` grew from 40 graphs to 45 in item
4 and to 58 in item 3. Item 3's additions are seven YUV grading chains of the
shape people actually write — `eq` and `hue` together, sometimes with `lutyuv`
or a `crop` between them — four graphs where an accepted filter is pinned to a
format it does not advertise, and three where a YUV option is inside the subset
but still rejected. The two `eq`/`hue`-on-`rgba` graphs moved out of the
"outside the pointwise subset" section, which is no longer why they are
refused, and `colorcontrast` took their place there.

```console
$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt
  graphs scanned:            58
  islands found:             50
  islands fusible today:     40
  frame passes eliminated:   36
  frame passes blocked:      9
  blocked passes by working format:
    negotiated   7 passes across 9 islands
    yuv410p      2 passes across 1 island

$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt --entry-format yuv420p
  frame passes eliminated:   39
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

Item 4 removed the format barrier and named the filter subset as the next
binding constraint. Item 3 has widened the subset on the YUV side; what binds
now is the RGB half, where `curves`, `colorbalance`, and `colorcontrast` need
an IR extension rather than a new table.

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

1. **A cross-channel float IR operation** (item 3's RGB half), which unlocks
   `colorbalance`, `colorcontrast`, and `curves` together rather than one at a
   time. This is now the binding constraint on reach. The sampling-group rule
   `hue` introduced is the guard it needs: such an operation reads all four
   channels, so it is admissible on every RGB layout and on `yuv444p`, and
   refused on `yuv422p` and `yuv420p`.
2. **`yuva*` support**, which is a bigger step than the other deferred table
   entries and worth separating from them. Alpha at full resolution alongside
   subsampled chroma gives a layout with three sampling groups rather than two,
   which is the first real exercise of the partition that `hue` made general.
3. **High-depth formats**: the widest change for the least corpus reach.
4. **The remaining YUV table entries** — `yuvj*`, `yuv411p`, `yuv410p`,
   `yuv440p` — if a real corpus ever asks for them. Each is a row in
   `layouts.py` plus an ABI identifier now that subsampling is general; only
   `yuv410p` appears in this corpus, and only as the "format the kernel cannot
   run" case.

Two entries left this list by being done. Layout-aware lowering landed as part
of item 4, because `negate`'s component mask could not be expressed in YUV
without it. `lutyuv`, `eq`, and `hue` landed as item 3's YUV half; the
cross-channel operation that was second is now first.
