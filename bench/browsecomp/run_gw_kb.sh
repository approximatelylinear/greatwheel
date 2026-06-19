#!/usr/bin/env bash
# Run the gw-kb evaluator against sample30, then score with quick_eval.
# Usage: ./run_gw_kb.sh [search-only|+topic|full]   (default: full)
#
# Spawns `gw-kb serve` in the background, polls /healthz, runs the
# OllamaAgent over sample30 with the matching system prompt + tools
# gating, then scores the resulting per-query JSON files.

set -euo pipefail

VARIANT=${1:-full}
case "$VARIANT" in
  search-only) PROMPT=prompts/gw_kb_search_only.txt ;;
  +topic)      PROMPT=prompts/gw_kb_with_topic.txt ;;
  full)        PROMPT=prompts/gw_kb_full.txt ;;
  *) echo "unknown variant: $VARIANT (expected: search-only, +topic, full)" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export DATABASE_URL=${DATABASE_URL:-postgres://gw:gw@localhost:5432/greatwheel_bc}
export GW_KB_LANCE_PATH=${GW_KB_LANCE_PATH:-data/kb-bc-lancedb}
export GW_KB_TANTIVY_PATH=${GW_KB_TANTIVY_PATH:-data/kb-bc-tantivy}
export PYO3_PYTHON=${PYO3_PYTHON:-$REPO_ROOT/crates/gw-kb/python/.venv/bin/python}
export GW_KB_PYTHON_PATH=${GW_KB_PYTHON_PATH:-$REPO_ROOT/crates/gw-kb/python}

# Activate the browsecomp venv (has requests, tqdm, etc.)
BROWSECOMP_VENV=${BROWSECOMP_VENV:-$REPO_ROOT/bench/browsecomp/.venv}
if [ -f "$BROWSECOMP_VENV/bin/activate" ]; then
  source "$BROWSECOMP_VENV/bin/activate"
fi

GW_KB_BIN=${GW_KB_BIN:-./target/release/gw-kb}
if [ ! -x "$GW_KB_BIN" ]; then
  GW_KB_BIN=./target/debug/gw-kb
fi
if [ ! -x "$GW_KB_BIN" ]; then
  echo "no gw-kb binary found; build with: cargo build --release -p gw-kb" >&2
  exit 1
fi

# 1. Kill any lingering gw-kb serve (avoids tantivy lock collision).
pkill -f "gw-kb serve" 2>/dev/null || true
sleep 2

# Start gw-kb serve in background.
"$GW_KB_BIN" serve --port 9099 > /tmp/gw_kb_serve.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT
echo "gw-kb serve PID=$SERVER_PID, waiting for health..."

for i in $(seq 1 60); do
  if curl -sf http://localhost:9099/healthz > /dev/null 2>&1; then
    echo "  ready after ${i}s"
    break
  fi
  sleep 1
done
if ! curl -sf http://localhost:9099/healthz > /dev/null 2>&1; then
  echo "gw-kb serve never came up; tail of log:" >&2
  tail -30 /tmp/gw_kb_serve.log >&2
  exit 1
fi

# 2. Warmup: first /search request triggers sentence-transformers load (~5s).
#    Doing it here keeps the first eval query honest.
curl -sf -X POST http://localhost:9099/search \
  -H 'content-type: application/json' \
  -d '{"query": "warmup", "k": 1}' > /dev/null || true
echo "warmup ok"

# 3. Run the agent across sample30
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=runs/gw-kb-${VARIANT}-${TS}
mkdir -p "$OUTPUT_DIR"
echo "writing runs to $OUTPUT_DIR"

cd bench/browsecomp
python3 ollama_client.py \
  --searcher-type gw-kb \
  --gw-kb-url http://localhost:9099 \
  --bc-jsonl "$REPO_ROOT/vendor/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl" \
  --gw-kb-tools "$VARIANT" \
  --query sample30.tsv \
  --output-dir "$REPO_ROOT/$OUTPUT_DIR" \
  --model qwen3.5:9b \
  --max-turns 12 \
  --k 10 \
  --include-get-document \
  --system-prompt-file "$PROMPT"

# 4. Score
cd "$REPO_ROOT"
python3 bench/browsecomp/quick_eval.py \
  --run-dir "$OUTPUT_DIR" \
  --fuzzy --llm-judge

echo
echo "done. results in $OUTPUT_DIR"
