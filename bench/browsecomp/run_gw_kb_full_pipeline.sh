#!/usr/bin/env bash
# End-to-end: tier 2 organize → link → classify → synthesize → eval ablations.
#
# Designed to be left running overnight. Each stage logs to its own file
# under /tmp/gw-kb-pipeline-<stage>.log so we can inspect them after.
# Stages are idempotent — re-running picks up where the last run stopped.
#
# Usage:
#   bench/browsecomp/run_gw_kb_full_pipeline.sh
#
# Env you can override:
#   DATABASE_URL, GW_KB_LANCE_PATH, GW_KB_TANTIVY_PATH, GW_KB_BIN
#
# Skip stages by setting SKIP_ORGANIZE=1, SKIP_LINK=1, SKIP_CLASSIFY=1,
# SKIP_SYNTH=1, SKIP_EVAL=1. Variants run after the KB is fully built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export DATABASE_URL=${DATABASE_URL:-postgres://gw:gw@localhost:5432/greatwheel_bc}
export GW_KB_LANCE_PATH=${GW_KB_LANCE_PATH:-data/kb-bc-lancedb}
export GW_KB_TANTIVY_PATH=${GW_KB_TANTIVY_PATH:-data/kb-bc-tantivy}
export PYO3_PYTHON=${PYO3_PYTHON:-$REPO_ROOT/crates/gw-kb/python/.venv/bin/python}
export GW_KB_PYTHON_PATH=${GW_KB_PYTHON_PATH:-$REPO_ROOT/crates/gw-kb/python}

GW_KB_BIN=${GW_KB_BIN:-./target/debug/gw-kb}
if [ ! -x "$GW_KB_BIN" ]; then
  echo "no gw-kb binary at $GW_KB_BIN — build first" >&2
  exit 1
fi

LOG_DIR=/tmp
PIPELINE_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "===== gw-kb full pipeline starting at $PIPELINE_TS ====="

# Reuse the running organize if any; otherwise start one and wait on its PID.
run_stage() {
  local name="$1" log="$2"; shift 2
  echo
  echo "----- [$(date -u +%H:%M:%SZ)] $name -----"
  echo "log: $log"
  if "$@" > "$log" 2>&1; then
    tail -3 "$log"
    echo "[$(date -u +%H:%M:%SZ)] $name OK"
  else
    echo "[$(date -u +%H:%M:%SZ)] $name FAILED, see $log" >&2
    tail -20 "$log" >&2
    exit 1
  fi
}

# ── Stage 1: organize ────────────────────────────────────────────────
if [ "${SKIP_ORGANIZE:-0}" != "1" ]; then
  RUNNING_ORGANIZE_PID=$(pgrep -f "$GW_KB_BIN organize" | head -1 || true)
  if [ -n "$RUNNING_ORGANIZE_PID" ]; then
    echo "----- [$(date -u +%H:%M:%SZ)] organize already running (PID $RUNNING_ORGANIZE_PID), waiting -----"
    while kill -0 "$RUNNING_ORGANIZE_PID" 2>/dev/null; do
      sleep 30
    done
    echo "[$(date -u +%H:%M:%SZ)] organize finished"
  else
    run_stage organize "$LOG_DIR/gw-kb-pipeline-organize.log" "$GW_KB_BIN" organize
  fi
fi

# ── Stage 2: link ────────────────────────────────────────────────────
if [ "${SKIP_LINK:-0}" != "1" ]; then
  run_stage link "$LOG_DIR/gw-kb-pipeline-link.log" "$GW_KB_BIN" link
fi

# ── Stage 3: classify (type the edges) ───────────────────────────────
if [ "${SKIP_CLASSIFY:-0}" != "1" ]; then
  run_stage classify "$LOG_DIR/gw-kb-pipeline-classify.log" "$GW_KB_BIN" classify
fi

# ── Stage 4: synthesize per-topic summaries ──────────────────────────
if [ "${SKIP_SYNTH:-0}" != "1" ]; then
  run_stage synthesize "$LOG_DIR/gw-kb-pipeline-synthesize.log" "$GW_KB_BIN" synthesize
fi

# Quick KB shape report
echo
echo "----- [$(date -u +%H:%M:%SZ)] KB state after build -----"
docker exec docker-postgres-1 psql -U gw -d greatwheel_bc -c "
SELECT
  (SELECT count(*) FROM kb_sources)                                AS sources,
  (SELECT count(*) FROM kb_chunks)                                 AS chunks,
  (SELECT count(DISTINCT chunk_id) FROM kb_topic_chunks)           AS tagged,
  (SELECT count(*) FROM kb_topics)                                 AS topics,
  (SELECT count(*) FROM kb_topic_links)                            AS edges,
  (SELECT count(*) FROM kb_topics WHERE summary IS NOT NULL)       AS summarized;
"

# ── Stage 5: ablation runs ────────────────────────────────────────────
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  for variant in search-only +topic full; do
    echo
    echo "----- [$(date -u +%H:%M:%SZ)] eval variant: $variant -----"
    if bash bench/browsecomp/run_gw_kb.sh "$variant"; then
      echo "[$(date -u +%H:%M:%SZ)] $variant OK"
    else
      echo "[$(date -u +%H:%M:%SZ)] $variant FAILED" >&2
    fi
  done
fi

echo
echo "===== pipeline complete at $(date -u +%Y%m%dT%H%M%SZ) ====="
echo "results dirs:"
ls -d runs/gw-kb-* 2>/dev/null || true
