#!/bin/bash
# Per-query driver that isolates ouros crashes.
# Each query runs in its own gw-bench process; a panic on one query
# does not kill the rest.
#
# Usage: run_per_query.sh <output-dir> <k> <max-turns> [extra gw-bench args...]
set -u

OUT_DIR="$1"
K="$2"
MAX_TURNS="$3"
shift 3
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"
TSV=bench/browsecomp/sample30.tsv
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

n=0
crashes=0
while IFS=$'\t' read -r qid qtext; do
    n=$((n+1))
    # Skip if already done (idempotent resume)
    if ls "$OUT_DIR"/run_"${qid}"_*.json >/dev/null 2>&1; then
        echo "[$n/30] q=$qid: already done, skipping"
        continue
    fi
    echo "[$n/30] q=$qid: running"
    if ! target/release/gw-bench \
        --search-backend http \
        --search-url http://localhost:8000 \
        --search-mode bm25 \
        --model qwen3.5:9b \
        --config bench/browsecomp/configs/baseline.toml \
        --query "$qtext" \
        --query-id "$qid" \
        --output-dir "$OUT_DIR" \
        --k "$K" \
        --max-turns "$MAX_TURNS" \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/q${qid}.log" 2>&1; then
        crashes=$((crashes+1))
        echo "  -> exit != 0 (possible ouros panic), continuing"
    fi
done < "$TSV"

echo ""
echo "DONE: $n queries attempted, $crashes crashes"
ls "$OUT_DIR"/run_*.json | wc -l | xargs echo "output files:"
