# FFmpeg fused-filter patches

## Week 5 dynamic kernel filter

`vf_fused.c` is the Week 5 integration filter. It loads the generated kernel
ABI, validates its version, selected pixel-format identifier, and plan hash,
then invokes the kernel in row slices through FFmpeg's worker executor. The
filter maps all 47 accepted packed and planar RGB/YUV layouts, from eight to
sixteen bits per component, and aligns slice boundaries to the chroma grid. It
also copies all frame properties and applies the compiler IR's requested
removal of color-dependent side data.

Formats above eight bits store little-endian 16-bit samples. The generated
kernel refuses a big-endian build rather than silently interpreting those
samples in native byte order.

The loader requires an absolute kernel path under an absolute, private trusted
root. Both must belong to the current user; the root cannot be group- or
world-writable and the library cannot be world-writable. The kernel must be a
direct child of that root after resolving symlinks.

Build the pinned patched fork from the repository root:

```sh
./scripts/build-ffmpeg-week5.sh
```

This creates a separate `.work/ffmpeg-week5` worktree and installs the binary
under `.build/ffmpeg-week5-<platform>`.

## Week 1 hand-fused experiment

`vf_fused_week1.c` is retained as the original experiment. It hardcodes
the selected `negate,lutrgb` candidate so Week 1 can test the cheapest decisive
hypothesis: whether one slice-threaded RGBA pass beats the two exact upstream
passes.

Create a separate worktree, register the filter, and build it without modifying
the pinned oracle tree:

```sh
git -C .work/ffmpeg worktree add ../ffmpeg-fused \
  38b88335f99e76ed89ff3c93f877fdefce736c13
./ffmpeg-patch/apply-week1-experiment.sh

(
  cd .work/ffmpeg-fused
  ./configure \
    --prefix="$PWD/../../.build/ffmpeg-fused-macos" \
    --cc=/usr/bin/clang --disable-doc --disable-ffplay --disable-stripping
  make -j"$(sysctl -n hw.logicalcpu)"
  make install
)
```

The filter always allocates a distinct RGBA output frame, copies frame
properties, mirrors `lutrgb`'s removal of color-dependent side data, advertises
`AVFILTER_FLAG_SLICE_THREADS`, and retains the 8-bit value between its negate
and LUT operations.
