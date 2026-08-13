#!/bin/bash
set -euo pipefail

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mode=${1:-screen}

case "$mode" in
    screen)
        warmups=${BENCH_WARMUPS:-1}
        runs=${BENCH_RUNS:-1}
        frames=${BENCH_FRAMES:-60}
        ;;
    full)
        warmups=${BENCH_WARMUPS:-1}
        runs=${BENCH_RUNS:-5}
        frames=${BENCH_FRAMES:-300}
        ;;
    *)
        echo "usage: $0 [screen|full]" >&2
        exit 2
        ;;
esac

case "$(uname -s)" in
    Darwin)
        platform=macos
        native_threads=$(sysctl -n hw.logicalcpu)
        ;;
    Linux)
        platform=linux
        native_threads=$(getconf _NPROCESSORS_ONLN)
        ;;
    *)
        echo "unsupported benchmark host: $(uname -s)" >&2
        exit 2
        ;;
esac

ffmpeg_bin=${FFMPEG_BIN:-"$project_root/.build/ffmpeg-$platform/bin/ffmpeg"}
chains_file=${CHAINS_FILE:-"$project_root/benchmarks/chains.tsv"}
resolutions=${BENCH_RESOLUTIONS:-"1920x1080 3840x2160"}
thread_counts=${BENCH_THREADS:-"1 $native_threads"}
selected_chains=${CHAIN_IDS:-all}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_dir=${RESULT_DIR:-"$project_root/benchmarks/results/$timestamp-$mode"}

if [ ! -x "$ffmpeg_bin" ]; then
    echo "pinned FFmpeg not found at $ffmpeg_bin; run scripts/build-ffmpeg.sh" >&2
    exit 2
fi

mkdir -p "$result_dir/logs"
csv="$result_dir/runs.csv"
printf '%s\n' 'kind,chain_id,stage_count,resolution,filter_threads,run_index,frames,user_seconds,system_seconds,wall_seconds,maxrss_kib,fps,exit_status,log_file' >"$csv"

{
    echo "mode=$mode"
    echo "utc_started=$timestamp"
    echo "ffmpeg_bin=$ffmpeg_bin"
    echo "ffmpeg_revision=38b88335f99e76ed89ff3c93f877fdefce736c13"
    echo "host_os=$(uname -s)"
    echo "host_release=$(uname -r)"
    echo "host_arch=$(uname -m)"
    echo "native_threads=$native_threads"
    echo "resolutions=$resolutions"
    echo "filter_threads=$thread_counts"
    echo "frames=$frames"
    echo "warmups=$warmups"
    echo "recorded_runs=$runs"
    echo "selected_chains=$selected_chains"
    echo
    "$ffmpeg_bin" -version
    echo
    "$ffmpeg_bin" -buildconf
} >"$result_dir/metadata.txt" 2>&1
cp "$chains_file" "$result_dir/chains.tsv"

is_selected() {
    case "$selected_chains" in
        all) return 0 ;;
        *) case ",$selected_chains," in *",$1,"*) return 0 ;; *) return 1 ;; esac ;;
    esac
}

run_one() {
    local kind=$1 chain_id=$2 stages=$3 resolution=$4 filter_threads=$5 run_index=$6 chain=$7
    local stem="${kind}-${chain_id}-${resolution}-t${filter_threads}-r${run_index}"
    local log="$result_dir/logs/$stem.log"
    local command_file="$result_dir/logs/$stem.command"
    local vf="format=rgba,${chain},format=rgba"
    local status=0 bench_line rss_line user_seconds system_seconds wall_seconds maxrss_reported maxrss_kib fps
    local -a command=(
        "$ffmpeg_bin" -hide_banner -nostdin -benchmark
        -f lavfi -i "testsrc2=size=${resolution}:rate=60"
        -frames:v "$frames" -filter_threads "$filter_threads"
        -vf "$vf" -an -f null -
    )

    printf '%q ' "${command[@]}" >"$command_file"
    printf '\n' >>"$command_file"
    "${command[@]}" >"$log" 2>&1 || status=$?

    bench_line=$(grep 'bench: utime=' "$log" | tail -n 1 || true)
    rss_line=$(grep 'bench: maxrss=' "$log" | tail -n 1 || true)
    user_seconds=$(printf '%s\n' "$bench_line" | sed -nE 's/.*utime=([0-9.]+)s.*/\1/p')
    system_seconds=$(printf '%s\n' "$bench_line" | sed -nE 's/.*stime=([0-9.]+)s.*/\1/p')
    wall_seconds=$(printf '%s\n' "$bench_line" | sed -nE 's/.*rtime=([0-9.]+)s.*/\1/p')
    maxrss_reported=$(printf '%s\n' "$rss_line" | sed -nE 's/.*maxrss=([0-9]+)KiB.*/\1/p')
    user_seconds=${user_seconds:-0}
    system_seconds=${system_seconds:-0}
    wall_seconds=${wall_seconds:-0}
    maxrss_reported=${maxrss_reported:-0}
    # FFmpeg's getmaxrss() treats ru_maxrss as KiB on every getrusage host.
    # Darwin actually returns bytes, so its printed number needs normalization.
    if [ "$(uname -s)" = Darwin ]; then
        maxrss_kib=$((maxrss_reported / 1024))
    else
        maxrss_kib=$maxrss_reported
    fi
    fps=$(awk -v frames="$frames" -v seconds="$wall_seconds" 'BEGIN { if (seconds > 0) printf "%.3f", frames / seconds; else print "0" }')

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$kind" "$chain_id" "$stages" "$resolution" "$filter_threads" \
        "$run_index" "$frames" "$user_seconds" "$system_seconds" \
        "$wall_seconds" "$maxrss_kib" "$fps" "$status" \
        "logs/$stem.log" >>"$csv"

    if [ "$status" -ne 0 ]; then
        echo "benchmark failed ($status): $chain_id $resolution threads=$filter_threads" >&2
        tail -n 20 "$log" >&2
        return "$status"
    fi
}

while IFS=$'\t' read -r chain_id stages description chain; do
    case "$chain_id" in ''|'#'*) continue ;; esac
    is_selected "$chain_id" || continue
    for resolution in $resolutions; do
        for filter_threads in $thread_counts; do
            i=1
            while [ "$i" -le "$warmups" ]; do
                echo "warmup $i/$warmups: $chain_id $resolution threads=$filter_threads"
                run_one warmup "$chain_id" "$stages" "$resolution" "$filter_threads" "$i" "$chain"
                i=$((i + 1))
            done
            i=1
            while [ "$i" -le "$runs" ]; do
                echo "run $i/$runs: $chain_id $resolution threads=$filter_threads"
                run_one recorded "$chain_id" "$stages" "$resolution" "$filter_threads" "$i" "$chain"
                i=$((i + 1))
            done
        done
    done
done <"$chains_file"

python3 "$project_root/benchmarks/harness/summarize.py" "$csv"
echo "results: $result_dir"
