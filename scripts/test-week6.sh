#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

if ! command -v "${LAVFI_CC_CLANG:-clang}" >/dev/null 2>&1; then
    echo "Week 6 requires Clang (set LAVFI_CC_CLANG to its path)" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) platform=macos ;;
    Linux) platform=linux ;;
    *) echo "unsupported Week 6 host: $(uname -s)" >&2; exit 2 ;;
esac

week6_ffmpeg=${LAVFI_CC_WEEK6_FFMPEG:-${LAVFI_CC_WEEK5_FFMPEG:-"$repo_dir/.build/ffmpeg-week5-$platform/bin/ffmpeg"}}
if [ ! -x "$week6_ffmpeg" ]; then
    echo "missing patched FFmpeg; run ./scripts/build-ffmpeg-week5.sh" >&2
    exit 1
fi

week6_cache=$(mktemp -d)
trap 'rm -rf "$week6_cache"' EXIT HUP INT TERM

LAVFI_CC_CACHE_DIR="$week6_cache/cache" \
LAVFI_CC_WEEK5_FFMPEG="$week6_ffmpeg" \
    python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/sanitize-week6.py
python3 scripts/benchmark-week6.py
