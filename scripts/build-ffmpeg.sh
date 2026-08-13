#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision=38b88335f99e76ed89ff3c93f877fdefce736c13
tag=n8.1.2
source_dir=${FFMPEG_SOURCE_DIR:-"$project_root/.work/ffmpeg"}

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
prefix=${FFMPEG_PREFIX:-"$project_root/.build/ffmpeg-$platform"}

if [ ! -d "$source_dir/.git" ]; then
    mkdir -p "$(dirname -- "$source_dir")"
    git clone --depth 1 --branch "$tag" \
        https://github.com/FFmpeg/FFmpeg.git "$source_dir"
fi

actual_revision=$(git -C "$source_dir" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
    echo "FFmpeg checkout is $actual_revision; expected $revision" >&2
    exit 2
fi

mkdir -p "$prefix"
record_only=${RECORD_ONLY:-0}
if [ "$record_only" != 1 ]; then
    (
        cd "$source_dir"
        ./configure \
            --prefix="$prefix" \
            --cc="$cc" \
            --disable-doc \
            --disable-ffplay \
            --disable-stripping
        make -j"$jobs"
        make install
    )
elif [ ! -x "$prefix/bin/ffmpeg" ]; then
    echo "cannot record missing build: $prefix/bin/ffmpeg" >&2
    exit 2
fi

manifest="$prefix/build-manifest.txt"
{
    echo "ffmpeg_tag=$tag"
    echo "ffmpeg_revision=$revision"
    echo "source_dir=$source_dir"
    echo "prefix=$prefix"
    echo "host_os=$(uname -s)"
    echo "host_arch=$(uname -m)"
    echo "logical_cpus=$jobs_default"
    echo "cc=$cc"
    echo
    "$cc" --version
    echo
    "$prefix/bin/ffmpeg" -version
    echo
    "$prefix/bin/ffmpeg" -buildconf
} >"$manifest" 2>&1

if [ "$record_only" = 1 ]; then
    echo "verified $prefix/bin/ffmpeg"
else
    echo "built $prefix/bin/ffmpeg"
fi
echo "recorded $manifest"
