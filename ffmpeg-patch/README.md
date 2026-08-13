# Week 1 hand-fused experiment

`vf_fused.c` is intentionally not the final dynamic-kernel filter. It hardcodes
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
