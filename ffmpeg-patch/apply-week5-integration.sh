#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_tree=${1:-"$project_root/.work/ffmpeg-week5"}
revision=38b88335f99e76ed89ff3c93f877fdefce736c13

if [ ! -d "$source_tree/.git" ] && [ ! -f "$source_tree/.git" ]; then
    echo "not an FFmpeg Git worktree: $source_tree" >&2
    exit 2
fi

actual_revision=$(git -C "$source_tree" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
    echo "FFmpeg checkout is $actual_revision; expected $revision" >&2
    exit 2
fi

if [ -e "$source_tree/libavfilter/vf_fused.c" ]; then
    echo "vf_fused.c already exists in $source_tree" >&2
    exit 2
fi

git -C "$source_tree" apply --check \
    "$project_root/ffmpeg-patch/series/0001-register-week1-fused-filter.patch"
git -C "$source_tree" apply \
    "$project_root/ffmpeg-patch/series/0001-register-week1-fused-filter.patch"
cp "$project_root/ffmpeg-patch/vf_fused.c" \
    "$source_tree/libavfilter/vf_fused.c"

echo "applied Week 5 fused filter integration to $source_tree"
