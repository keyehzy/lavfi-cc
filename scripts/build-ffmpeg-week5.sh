#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision=38b88335f99e76ed89ff3c93f877fdefce736c13
oracle_tree="$project_root/.work/ffmpeg"
source_tree=${FFMPEG_WEEK5_SOURCE_DIR:-"$project_root/.work/ffmpeg-week5"}

case "$(uname -s)" in
    Darwin)
        platform=macos
        cc_default=/usr/bin/clang
        jobs_default=$(sysctl -n hw.logicalcpu)
        ;;
    Linux)
        platform=linux
        cc_default=clang
        jobs_default=$(getconf _NPROCESSORS_ONLN)
        ;;
    *)
        echo "unsupported build host: $(uname -s)" >&2
        exit 2
        ;;
esac

cc=${CC:-$cc_default}
jobs=${BUILD_JOBS:-$jobs_default}
prefix=${FFMPEG_WEEK5_PREFIX:-"$project_root/.build/ffmpeg-week5-$platform"}

if [ ! -d "$oracle_tree/.git" ]; then
    echo "missing pinned oracle tree; run scripts/build-ffmpeg.sh first" >&2
    exit 2
fi
if [ ! -e "$source_tree/.git" ]; then
    git -C "$oracle_tree" worktree add "$source_tree" "$revision"
    "$project_root/ffmpeg-patch/apply-week5-integration.sh" "$source_tree"
fi
if [ ! -f "$source_tree/libavfilter/vf_fused.c" ]; then
    echo "Week 5 source tree is not patched: $source_tree" >&2
    exit 2
fi

mkdir -p "$prefix"
(
    cd "$source_tree"
    ./configure \
        --prefix="$prefix" \
        --cc="$cc" \
        --disable-doc \
        --disable-ffplay \
        --disable-stripping
    make -j"$jobs"
    make install
)

manifest="$prefix/build-manifest.txt"
{
    echo "ffmpeg_revision=$revision"
    echo "source_tree=$source_tree"
    echo "prefix=$prefix"
    echo "host_os=$(uname -s)"
    echo "host_arch=$(uname -m)"
    echo "cc=$cc"
    echo
    "$cc" --version
    echo
    "$prefix/bin/ffmpeg" -version
    echo
    "$prefix/bin/ffmpeg" -hide_banner -h filter=fused
} >"$manifest" 2>&1

echo "built Week 5 FFmpeg at $prefix/bin/ffmpeg"
echo "recorded $manifest"
