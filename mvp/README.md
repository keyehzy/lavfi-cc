# Shutter Encoder color-grading MVP

> Accelerated CPU color grading for Shutter Encoder exports.

This directory is the application-specific boundary around `lavfi-cc`. It
accepts the linear video graph Shutter emits with `-filter_complex`, verifies
the graph's negotiated formats with the pinned FFmpeg build, and fuses every
profitable compatible grading island. FFmpeg still handles decoding, encoding,
audio, format conversion, and every filter outside the compiler subset.

## Supported integration shape

The execution path accepts exactly one graph shaped like:

```text
[0:v]<one comma-separated video chain>[out]
```

Several chains, branches, internal labels, and a combined audio graph fail
closed to ordinary FFmpeg. This is intentional: the adapter does not reinterpret
general `-filter_complex` syntax.

Before lowering, the adapter normalizes the positional forms emitted by the
pinned Shutter source:

```text
colortemperature=5200:pl=1
vibrance=.4:rbal=1.1:gbal=.9:bbal=1
```

to:

```text
colortemperature=temperature=5200:pl=1
vibrance=intensity=.4:rbal=1.1:gbal=.9:bbal=1
```

For each maximal run with a common supported format, a one-frame replay adds a
uniquely named `showinfo` stage. A format boundary is inserted only if pinned
FFmpeg reports a native format accepted by every filter in that run. This
turns an implicit negotiation decision into an explicit, checked compiler
input. The execution path reuses the export's real input options, video codec,
pixel format, and profile, then replaces the destination with a null muxer so
the probe cannot overwrite an export. `explain` has no export command, so it
uses a synthetic `testsrc2` input and reports that narrower probe scope.
Unsupported filters are copied unchanged and split islands safely.

The default FFmpeg version is `n8.1.2`. A mismatch falls back to the exact
original command, or fails when `--require-fusion` is used.

## Explain the demonstrated graph

`shutter-demo.filtergraph` is the exact six-stage graph for Highlights +25,
Midtones +20, Shadows +25, Whites +20, Blacks -10, and white balance 5200 K.

```sh
./mvp/lavfi-cc-shutter explain \
  --ffmpeg .build/ffmpeg-week5-linux/bin/ffmpeg \
  --filter-complex "$(<mvp/shutter-demo.filtergraph)"
```

Use `--json` for the normalized graph, verified pins, island plans, hashes, and
placeholder rewrite.

## Run an export command

Keep Shutter's FFmpeg arguments intact after `--`:

```sh
./mvp/lavfi-cc-shutter run \
  --ffmpeg /path/to/pinned/patched/ffmpeg \
  --require-fusion -- \
  -i input.mp4 \
  -filter_complex "[0:v]<Shutter color graph>[out]" \
  -map "[out]" -map 0:a? -c:v libx264 output.mp4
```

Without `--require-fusion`, parse, probe, compile, cache, or preflight failures
execute the original arguments through the configured FFmpeg binary. Neither
normalization nor format pins leak into fallback.

For a Shutter test installation that can override its FFmpeg executable,
`mvp/shutter-ffmpeg` is a drop-in proxy. Point `LAVFI_CC_FFMPEG` at the real
pinned and patched binary, then configure the test installation to invoke the
proxy. Version checks and commands without `-filter_complex` pass through
silently; export graphs use the same safe-fallback behavior.

```sh
export LAVFI_CC_FFMPEG=/absolute/path/to/pinned/patched/ffmpeg
export LAVFI_CC_SHUTTER_REQUIRE_FUSION=1  # recommended for benchmark runs
/absolute/path/to/repository/mvp/shutter-ffmpeg -version
```

## Correctness and credibility gates

Run the focused unit tests with:

```sh
python3 -m unittest tests.test_shutter_mvp
```

The repository-local macOS check uses real pinned FFmpeg, a compiled `rgb24`
kernel, and raw output comparison, but it is not the release benchmark. The
authoritative gate remains five or more warm runs launched from commands
captured from the actual Shutter UI, for both `libx264` H.264 and ProRes, on
native Linux x86-64. Retain the UI settings, complete command, FFmpeg build
identity, input digest, per-run logs, encoded hashes, decoded-frame hashes, and
median wall times in `benchmarks/results/`.

`mvp.benchmark` automates that replay and refuses to label other hosts as
authoritative. Copy the FFmpeg invocation from Shutter's console into a file,
replace its one output path with `{output}` while retaining the container
suffix (for example `{output}.mp4`), then run one gate per codec:

```sh
python3 -m mvp.benchmark \
  --command-file shutter-h264.command \
  --ffmpeg .build/ffmpeg-week5-linux/bin/ffmpeg \
  --codec h264 --samples 5

python3 -m mvp.benchmark \
  --command-file shutter-prores.command \
  --ffmpeg .build/ffmpeg-week5-linux/bin/ffmpeg \
  --codec prores --samples 5
```

The harness alternates baseline and warm-cache fused runs, retains every exact
command and log, hashes both the encoded files and decoded frame streams, and
writes medians and speedup to `summary.md` in its timestamped result directory.
