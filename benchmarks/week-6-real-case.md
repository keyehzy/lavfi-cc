# Week 6 directional decode-filter-encode benchmark

Date: 2026-08-13  
lavfi-cc commit: `d2e6dbc99a194fbf8e47f01ca8fb4217d0cb3cde`

## Result

On the macOS arm64 development host, the warm-cache fused path reduced median
wall time from 3.138 seconds to 2.276 seconds for a 30-second 1080p segment
decoded from H.264, passed through the four-filter balanced chain, and encoded
to an actual MP4 file.

| case | samples | median wall | median throughput | wall reduction | speedup |
|---|---:|---:|---:|---:|---:|
| ordinary FFmpeg chain | 5 | 3.138 s | 239.0 fps | — | 1.000x |
| fused, warm cache | 5 | 2.276 s | 329.5 fps | 27.5% | 1.379x |

Throughput improved by 37.9%. This directionally exceeds the MVP requirement
that at least one decode-filter-encode workload improve by 15% when filtering
is a material part of runtime. It is not the authoritative acceptance result:
that measurement must be repeated on native Linux x86-64.

## Environment

- Host: Apple M4, 10 logical CPUs, 16 GiB RAM.
- OS: macOS 15.7.7.
- FFmpeg: pinned `n8.1.2` patched with the Week 5 `fused` AVFilter.
- Compiler: Apple Clang 17.0.0.
- Input: `video.mp4`, H.264 `yuv420p`, 1920x1080, 25 fps, 213.09 seconds.
- Input SHA-256:
  `86fd7c5e738150e84b13b20d568caec8c25d77a2be3259a4c907a9a0ac83efaa`.
- Treatment: the 30-second segment beginning at 30 seconds, 750 video frames;
  audio excluded.
- Output: software MPEG-4 Part 2, quality 5, four encoder threads, MP4 written
  to local temporary storage.
- Filter execution: ten filter threads.

The bounded filtergraph was:

```text
format=rgba,
negate,
lutrgb=r=val*1.08+2:g=val*0.94+4:b=val*0.88+12:a=val,
colorlevels=rimin=0.04:gimin=0.02:bimin=0.06:rimax=0.96:gimax=0.98:bimax=0.94:preserve=none,
colorchannelmixer=rr=0.90:rg=0.08:rb=0.02:gr=0.03:gg=0.94:gb=0.03:br=0.04:bg=0.06:bb=0.90:pc=none,
format=yuv420p
```

The cache was populated before timing. Baseline and fused treatments each
received one discarded warm-up. Five recorded samples per case were then
alternated in this order to reduce ordering and thermal bias:

```text
baseline, fused, fused, baseline, baseline,
fused, fused, baseline, baseline, fused
```

Wall time was measured outside the process with a monotonic clock. Fused wall
time therefore includes Python wrapper startup, cache lookup and validation,
the synthetic fused-filter preflight process, the main FFmpeg process, MP4
muxing, and local output writes.

## Raw MP4 samples

| order | case | wall (s) | fps | output bytes | SHA-256 prefix |
|---:|---|---:|---:|---:|---|
| 1 | baseline | 3.094 | 242.4 | 20,477,409 | `6ea1ece823d1` |
| 2 | fused | 2.276 | 329.5 | 20,477,409 | `6ea1ece823d1` |
| 3 | fused | 2.271 | 330.2 | 20,477,409 | `6ea1ece823d1` |
| 4 | baseline | 3.138 | 239.0 | 20,477,409 | `6ea1ece823d1` |
| 5 | baseline | 3.112 | 241.0 | 20,477,409 | `6ea1ece823d1` |
| 6 | fused | 2.273 | 330.0 | 20,477,409 | `6ea1ece823d1` |
| 7 | fused | 2.290 | 327.5 | 20,477,409 | `6ea1ece823d1` |
| 8 | baseline | 3.165 | 237.0 | 20,477,409 | `6ea1ece823d1` |
| 9 | baseline | 3.175 | 236.2 | 20,477,409 | `6ea1ece823d1` |
| 10 | fused | 2.330 | 321.8 | 20,477,409 | `6ea1ece823d1` |

All ten encoded MP4 files had the same size and full SHA-256 during the run.
Only this digest prefix was retained in the captured output:

```text
6ea1ece823d1
```

The temporary MP4 files were removed after comparison, so this report does not
claim an unrecovered full digest. A separate one-second rawvideo hash check
compared the complete baseline and fused decoded outputs and matched exactly:

```text
SHA256=484b4b8fc42ed1a9b948f67cf30ae585c6f97968d7976c35dee491d3353faa57
```

## Week 6 cache contribution

A second five-sample experiment used the same workload but sent encoded
packets to the null muxer. Each cold treatment received a new empty cache;
the warm treatment reused one populated cache. The software encoder still ran
in both treatments.

| fused cache state | wall samples (s) | median wall | median fps |
|---|---|---:|---:|
| fresh miss every run | 2.786, 2.696, 2.641, 2.635, 2.658 | 2.658 s | 282.2 |
| warm hit | 2.312, 2.537, 2.313, 2.318, 2.540 | 2.318 s | 323.6 |

Warm caching reduced fused wall time by 12.8%, a 1.147x speedup over compiling
on every invocation. Against the 3.197-second ordinary-chain median in that
experiment, the cold-miss fused path improved throughput by 20.2%, while the
warm-cache path improved it by 37.9%.

## Interpretation and follow-up

The result supports two separate conclusions:

1. Fusing the four pointwise filters materially improves this end-to-end
   workload; all fused samples were faster than all baseline samples.
2. The Week 6 cache makes repeated real jobs measurably faster by removing most
   compilation startup from every invocation.

The test uses a real decoder, encoder, muxer, and output file, but it remains a
directional development-host result. Before using it as the final MVP claim:

- repeat at least five warm samples on Linux x86-64;
- use a representative production codec such as libx264 or libx265;
- add longer clips and report break-even frame counts;
- separately run the Week 7 filter-only 4K performance gate; and
- retain commands, full output hashes, host metadata, and raw logs in an
  archived result directory.
