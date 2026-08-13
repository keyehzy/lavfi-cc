# Reach roadmap status

The six items below were the planned follow-up to the Week 6 MVP. This
document records what landed, what did not, and — where something did not — the
specific finding that determines how much work is left.

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
With YUV supported, the second is now the binding constraint.

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

## 3. More pointwise filters — not started, with one finding that reshapes it

No new filters landed. The lowering layer was extracted into
`lavfi_cc/filters.py` so adding one is now local, but the filters named in the
roadmap split into two groups that need different prerequisites:

- `lutyuv`, `eq`, and `hue` are YUV-native and were blocked on item 4. That
  block is gone: `yuv444p`, `yuv422p`, and `yuv420p` islands now fuse in place,
  and `lutyuv` in particular is channel-independent, so it fits the subsampled
  plane walk without further IR work. They are the first thing to do next.
- `curves`, `colorbalance`, and `colorcontrast` are RGB-native but are **not
  channel-independent**, so they do not fit the current IR at all.

`colorbalance` is the clearest example. Its per-pixel lightness term couples the
three channels:

```c
const float l = (FFMAX3(r, g, b) + FFMIN3(r, g, b));
r = get_component(r, l, shadows, midtones, highlights);
```

Neither a per-channel `lut8` nor a `matrix4x4` can express that. It is still
pixel-local and therefore fusible in principle, but it needs a new IR operation
evaluating a fixed float32 expression across all four channels, with upstream's
exact operation order, `av_clipf` clamping, and `lrintf` ties-to-even rounding
reproduced. That is a genuine IR extension, not a new table.

Adding a filter whose semantics are only approximately right would be worse
than not adding it, because bit-exactness against the pinned oracle is the
property this compiler exists to preserve.

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
an operation could mix, and a subsampled layout accepts only channel-independent
operations. `lut8` and the diagonal `levels_f32_fma` qualify; a
`colorchannelmixer` matrix does not. `validate_ir` rejects the combination, so
the rule reaches the interpreter, the optimizer, and code generation alike and
cannot be dodged by constructing the IR directly.

Both the interpreter and the code generator therefore have two shapes: a
whole-pixel walk for a layout whose planes share the frame's resolution — which
is every RGB layout, plus `yuv444p` — and a per-plane walk at each plane's own
resolution for `yuv422p` and `yuv420p`. Generated C for the existing layouts is
unchanged, so no cached or bundled kernel was invalidated.

### The layout reached the lowering, and the option names moved with it

`negate` is the only accepted filter that advertises YUV at all; `lutrgb`,
`colorlevels`, and `colorchannelmixer` are RGB-only upstream, so a YUV run
containing one of them is refused rather than fused. That refusal is not a
detail: the corpus graph `format=yuv420p,negate,lutrgb=…,colorlevels=…` used to
be reported as a two-pass island waiting on YUV support, and it never was one —
FFmpeg converts around `lutrgb`, so those two passes were never removable.
Supporting YUV is what made that visible.

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

Generated C for the pre-existing layouts is byte-identical to before this work
across all 40 layout-and-chain combinations, so no cached or bundled kernel was
invalidated.

**Deferred:** `yuva*` (alpha at full resolution alongside subsampled chroma),
the `yuvj` full-range family, and the remaining ratios `yuv411p`, `yuv410p`,
and `yuv440p`. Each is a table entry rather than a design question, but none of
them appears in the corpus.

## 5. Analysis-only scanner — done

`lavfi-cc scan` consumes arbitrary filtergraphs — several chains, link labels,
whitespace, hardware filters, options this compiler cannot parse — and never
compiles or runs anything. A lenient parser records per-filter option problems
instead of failing the graph, so a scan still explains what blocked an island.

Against the corpus in `tests/corpus/filtergraphs.txt`, which item 4 grew from
40 graphs to 45: four YUV-native grading chains, since decoded video arrives in
`yuv420p` and the corpus had almost no such graph, plus one pinned to
`yuv410p`, a ratio that is still unimplemented and now carries the "format the
kernel cannot run" case that `yuv420p` used to.

```console
$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt
  graphs scanned:            45
  islands found:             36
  islands fusible today:     28
  frame passes eliminated:   27
  frame passes blocked:      8
  blocked passes by working format:
    negotiated   6 passes across 7 islands
    yuv410p      2 passes across 1 island

$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt --entry-format yuv420p
  frame passes eliminated:   29
  frame passes blocked:      4
```

Blockers are ranked by the frame passes they withhold rather than by how often
they appear, which is what makes the output actionable.

Item 2's format work moved this corpus from 17 eliminated passes to 19 with
`rgb24`, then to 21 with planar `gbrp`, dropping blocked passes from 12 to 8.

Item 4's effect is best read on the corpus as it stood *before* those five new
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

The gain appears under `--entry-format yuv420p`, which is what a caller can
prove for decoded video: `fps=30,negate,negate,negate` is a genuine YUV island
that a kernel now replaces.

That is the honest shape of the result, and it names the next blocker. `negate`
is the only accepted filter that runs in YUV, so until item 3's `lutyuv`, `eq`,
and `hue` land, a YUV island can only be built out of `negate`s. Item 4 removed
the format barrier; the filter subset is now the binding one.

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

1. **`lutyuv`, `eq`, and `hue`** (item 3's YUV half), now unblocked. This is
   what turns item 4's format support into reach: `negate` alone can only build
   a YUV island out of more `negate`s, so every YUV grading chain in the wild
   still falls outside the subset. `lutyuv` is the cheapest of the three — it
   is `lutrgb`'s own source file with a different component vocabulary and a
   16–235/16–240 clamp on `clipval` and `negval` — and it is channel-independent,
   so it drops into the subsampled plane walk unchanged.
2. **A cross-channel float IR operation** (item 3's RGB half), which unlocks
   `colorbalance`, `colorcontrast`, and `curves` together rather than one at a
   time. Note that such an operation is *not* expressible on a subsampled
   layout, so it will need the guard `validate_ir` already applies.
3. **High-depth formats**: the widest change for the least corpus reach.
4. **The remaining YUV table entries** — `yuva*`, `yuvj*`, `yuv411p`,
   `yuv410p`, `yuv440p` — if a real corpus ever asks for them. Each is a row in
   `layouts.py` plus an ABI identifier now that subsampling is general, but none
   appears in this corpus.

Layout-aware lowering, previously third on this list, landed as part of item 4
because `negate`'s component mask could not be expressed in YUV without it.
