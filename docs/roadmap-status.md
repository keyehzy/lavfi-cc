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
next to item 1 (islands): they are the only way item 1's reach can grow. Island
discovery finds the runs; format coverage decides how many of them may actually
be fused.

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

## 2. Widen pixel-format reach — done for packed 8-bit, deferred above that

Supported today: `rgba`, `bgra`, `argb`, `abgr`, `rgb24`, `bgr24`.

The byte offsets in `lavfi_cc/layouts.py` were taken from the pinned binary
rather than from format descriptors, by converting one known `0x11223344` pixel
into each format and reading the bytes back.

The reduction that makes this cheap: the operations are layout-independent, so
only the load and store ends change. An alpha-less layout loads `a = 0` and
stores only three components, which is exactly what upstream does — the packed
`colorchannelmixer` omits every alpha term when `have_alpha` is unset, and the
other accepted filters treat channels independently, so an alpha lane that is
never stored cannot affect the stored ones.

Because all six are one plane addressed by one stride, the kernel ABI did not
change; only new format identifiers were added, and `vf_fused.c` now negotiates
the single format its loaded kernel was built for.

Two format-dependent constraints were found and are enforced:

- A run may only be fused in a format that *every* filter in it advertises,
  otherwise FFmpeg converts in the middle of the run and one kernel is no longer
  equivalent to the filters it replaced. `colorlevels` and `colorchannelmixer`
  accept the `0rgb`/`rgb0` family that `negate` and `lutrgb` do not, so that
  family is outside the common subset.
- `negate=components=…a` on an alpha-less format is a hard configuration
  failure upstream, not a silent no-op, so the compiler rejects it too.

**Deferred:** planar `gbrp`/`gbrap` and the 9–16-bit formats. Both need the
kernel ABI to carry per-plane pointers and strides instead of one plane, which
is the same ABI change item 4 needs. High-depth additionally needs 65536-entry
tables and a re-derivation of every quantization rule at that depth.

## 3. More pointwise filters — not started, with one finding that reshapes it

No new filters landed. The lowering layer was extracted into
`lavfi_cc/filters.py` so adding one is now local, but the filters named in the
roadmap split into two groups that need different prerequisites:

- `lutyuv`, `eq`, and `hue` are YUV-native and are blocked on item 4.
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

## 4. Avoid boundary conversion — not started, design settled

The roadmap offered two routes: fuse the format conversion into the island
boundary, or introduce a YUV IR. The first should not be attempted. Fusing the
conversion means reproducing swscale byte-exactly, and swscale selects among
many code paths; matching all of them is a far larger commitment than the
filters themselves.

The second route is the right one, and it is better described as *avoiding*
conversion rather than fusing it: support planar YUV natively so a YUV-native
island fuses in place with no conversion at either boundary, which is what
FFmpeg already does. Bit-exactness is then against the filters, which is
tractable, instead of against swscale, which is not.

Prerequisites: the plane-pointer ABI change shared with item 2's planar work,
plus chroma subsampling in the IR (a `yuv420p` island applies its Y mapping at
full resolution and its U/V mappings at half, so "one pixel" is no longer one
loop iteration).

The scanner already quantifies why this ranks first — see below.

## 5. Analysis-only scanner — done

`lavfi-cc scan` consumes arbitrary filtergraphs — several chains, link labels,
whitespace, hardware filters, options this compiler cannot parse — and never
compiles or runs anything. A lenient parser records per-filter option problems
instead of failing the graph, so a scan still explains what blocked an island.

Against the 40-graph corpus in `tests/corpus/filtergraphs.txt`:

```console
$ ./lavfi-cc scan --file tests/corpus/filtergraphs.txt
  graphs scanned:            40
  islands found:             29
  islands fusible today:     21
  frame passes eliminated:   19
  frame passes blocked:      10
  blocked passes by working format:
    negotiated   6 passes across 6 islands
    yuv420p      2 passes across 1 island
    gbrp         2 passes across 1 island
```

Blockers are ranked by the frame passes they withhold rather than by how often
they appear, which is what makes the output actionable. On this corpus the
largest single blocker is islands whose format negotiation decides — the case
item 4 addresses — followed by planar and YUV formats. That is the evidence
behind the ordering recommended at the end of this document.

Adding `rgb24` support in item 2 moved this corpus from 17 to 19 eliminated
passes and dropped blocked passes from 12 to 10.

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

1. **Plane-pointer kernel ABI.** One change unblocks planar `gbrp`/`gbrap`
   (item 2) and all of item 4. Nothing else in items 2–4 can proceed far
   without it.
2. **Planar YUV islands** (item 4), including subsampling. The scanner says
   negotiated and YUV islands hold the most blocked passes, and this is what
   makes `lutyuv`, `eq`, and `hue` reachable.
3. **A cross-channel float IR operation** (item 3), which unlocks
   `colorbalance`, `colorcontrast`, and `curves` together rather than one at a
   time.
4. **High-depth formats**, last: the widest change for the least corpus reach.
