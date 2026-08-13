#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

if ! command -v "${LAVFI_CC_CLANG:-clang}" >/dev/null 2>&1; then
    echo "Week 5 requires Clang (set LAVFI_CC_CLANG to its path)" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) platform=macos ;;
    Linux) platform=linux ;;
    *) echo "unsupported Week 5 host: $(uname -s)" >&2; exit 2 ;;
esac

week5_ffmpeg=${LAVFI_CC_WEEK5_FFMPEG:-"$repo_dir/.build/ffmpeg-week5-$platform/bin/ffmpeg"}
if [ ! -x "$week5_ffmpeg" ]; then
    echo "missing Week 5 FFmpeg; run ./scripts/build-ffmpeg-week5.sh" >&2
    exit 1
fi

LAVFI_CC_WEEK5_FFMPEG="$week5_ffmpeg" \
    python3 -m unittest discover -s tests -p 'test_*.py' -v
