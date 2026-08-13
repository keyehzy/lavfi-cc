#!/bin/bash
set -euo pipefail

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

case "$(uname -s)" in
    Darwin) platform=macos ;;
    Linux) platform=linux ;;
    *) echo "unsupported corpus host: $(uname -s)" >&2; exit 2 ;;
esac

ffmpeg_bin=${FFMPEG_BIN:-"$project_root/.build/ffmpeg-$platform/bin/ffmpeg"}
cases_file=${CORPUS_CASES:-"$project_root/tests/corpus/cases.tsv"}
revision=38b88335f99e76ed89ff3c93f877fdefce736c13
namespace="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-${revision:0:12}"
output_dir=${CORPUS_OUTPUT_DIR:-"$project_root/tests/corpus/baseline/$namespace"}

if [ ! -x "$ffmpeg_bin" ]; then
    echo "pinned FFmpeg not found at $ffmpeg_bin; run scripts/build-ffmpeg.sh" >&2
    exit 2
fi

mkdir -p "$output_dir/commands"
manifest="$output_dir/manifest.tsv"
printf '%s\n' $'case_id\twidth\theight\tpattern\tbytes\tsha256\tfile\tfilter_chain' >"$manifest"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

source_for() {
    local pattern=$1 width=$2 height=$3
    case "$pattern" in
        black) printf 'color=c=black:size=%sx%s:rate=1' "$width" "$height" ;;
        white) printf 'color=c=white:size=%sx%s:rate=1' "$width" "$height" ;;
        testsrc) printf 'testsrc=size=%sx%s:rate=1' "$width" "$height" ;;
        testsrc2) printf 'testsrc2=size=%sx%s:rate=1' "$width" "$height" ;;
        rgba_ramp)
            printf "nullsrc=size=%sx%s:rate=1,format=rgba,geq=r='X/W*255':g='Y/H*255':b='(X+Y)/(W+H)*255':a='X/W*255',format=rgba" "$width" "$height"
            ;;
        checker)
            printf "nullsrc=size=%sx%s:rate=1,format=rgba,geq=r='mod(X*37+Y*17,256)':g='mod(X*11+Y*43,256)':b='mod(X*29+Y*7,256)':a='mod(X*13+Y*19,256)',format=rgba" "$width" "$height"
            ;;
        *) echo "unknown source pattern: $pattern" >&2; return 2 ;;
    esac
}

while IFS=$'\t' read -r case_id width height pattern chain; do
    case "$case_id" in ''|'#'*) continue ;; esac
    source_graph=$(source_for "$pattern" "$width" "$height")
    vf="format=rgba,${chain},format=rgba"
    output="$output_dir/$case_id.rgba"
    command_file="$output_dir/commands/$case_id.command"
    log="$output_dir/commands/$case_id.log"
    command=(
        "$ffmpeg_bin" -hide_banner -nostdin -v error
        -f lavfi -i "$source_graph" -frames:v 1
        -filter_threads 1 -vf "$vf" -an
        -pix_fmt rgba -c:v rawvideo -f rawvideo -y "$output"
    )

    printf '%q ' "${command[@]}" >"$command_file"
    printf '\n' >>"$command_file"
    "${command[@]}" >"$log" 2>&1

    bytes=$(wc -c <"$output" | tr -d ' ')
    expected_bytes=$((width * height * 4))
    if [ "$bytes" -ne "$expected_bytes" ]; then
        echo "$case_id produced $bytes bytes; expected $expected_bytes" >&2
        exit 1
    fi
    digest=$(sha256_file "$output")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_id" "$width" "$height" "$pattern" "$bytes" "$digest" \
        "$case_id.rgba" "$chain" >>"$manifest"
    echo "$case_id $digest"
done <"$cases_file"

{
    echo "ffmpeg_revision=$revision"
    echo "ffmpeg_bin=$ffmpeg_bin"
    echo "host_os=$(uname -s)"
    echo "host_release=$(uname -r)"
    echo "host_arch=$(uname -m)"
    echo "pixel_format=rgba"
    echo "frames_per_case=1"
    echo "filter_threads=1"
    echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    "$ffmpeg_bin" -version
    echo
    "$ffmpeg_bin" -buildconf
} >"$output_dir/metadata.txt" 2>&1

echo "corpus: $output_dir"
