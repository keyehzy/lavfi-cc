#!/bin/bash
set -euo pipefail

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle=${FFMPEG_ORACLE:-"$project_root/.build/ffmpeg-macos/bin/ffmpeg"}
fused=${FFMPEG_FUSED:-"$project_root/.build/ffmpeg-fused-macos/bin/ffmpeg"}
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/lavfi-cc-fused.XXXXXX")
trap 'rm -r "$work_dir"' EXIT

if [ ! -x "$oracle" ] || [ ! -x "$fused" ]; then
    echo "oracle or fused FFmpeg binary is missing" >&2
    exit 2
fi

baseline='format=rgba,negate,lutrgb=r=val*1.08+2:g=val*0.94+4:b=val*0.88+12:a=val,format=rgba'
replacement='format=rgba,fused,format=rgba'
widths='1 2 3 7 8 15 16 17 63 64 65 1919 1920'

for width in $widths; do
    height=$((width % 7 + 1))
    source="nullsrc=size=${width}x${height}:rate=1,format=rgba,geq=r='mod(X*37+Y*17,256)':g='mod(X*11+Y*43,256)':b='mod(X*29+Y*7,256)':a='mod(X*13+Y*19,256)',format=rgba"
    expected="$work_dir/$width.expected.rgba"
    actual="$work_dir/$width.actual.rgba"

    "$oracle" -hide_banner -nostdin -v error -f lavfi -i "$source" \
        -frames:v 1 -filter_threads 1 -vf "$baseline" -pix_fmt rgba \
        -c:v rawvideo -f rawvideo -y "$expected"
    "$fused" -hide_banner -nostdin -v error -f lavfi -i "$source" \
        -frames:v 1 -filter_threads 1 -vf "$replacement" -pix_fmt rgba \
        -c:v rawvideo -f rawvideo -y "$actual"

    if ! cmp -s "$expected" "$actual"; then
        echo "pixel mismatch at width=$width height=$height" >&2
        cmp -l "$expected" "$actual" | head -n 1 >&2 || true
        exit 1
    fi
    echo "bit-exact width=$width height=$height"
done

echo "hand-fused output matches the pinned oracle"
