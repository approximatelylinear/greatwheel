#!/usr/bin/env bash
#
# Greatwheel BrowseComp-Plus benchmark runner
#
# Uses uv for Python dependency management.
#
# Usage:
#   # Setup (first time only)
#   ./bench/browsecomp/run.sh setup
#
#   # Run with BM25 searcher (needs Java 21 + Lucene index)
#   ./bench/browsecomp/run.sh bm25
#
#   # Run with LanceDB searcher (needs Ollama embeddings + built index)
#   ./bench/browsecomp/run.sh lancedb
#
#   # Evaluate results
#   ./bench/browsecomp/run.sh eval <run_dir>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENDOR_DIR="$PROJECT_ROOT/vendor/BrowseComp-Plus"
DATA_DIR="$VENDOR_DIR/data"
VENV_DIR="$SCRIPT_DIR/.venv"

# Defaults (override via env)
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
GW_MODEL="${GW_MODEL:-qwen2.5:7b}"
GW_EMBED_MODEL="${GW_EMBED_MODEL:-nomic-embed-text}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating venv at $VENV_DIR ..."
        uv venv "$VENV_DIR" --python 3.12
    fi
}

# Run a python command inside the bench venv via uv
uvrun() {
    uv run --project "$SCRIPT_DIR" "$@"
}

# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

cmd_setup() {
    echo "=== Setting up BrowseComp-Plus ==="

    # Create venv and sync base dependencies
    ensure_venv
    echo "Syncing Python dependencies ..."
    uv sync --project "$SCRIPT_DIR"

    # Decrypt the dataset
    if [ ! -f "$DATA_DIR/browsecomp_plus_decrypted.jsonl" ]; then
        echo "Decrypting dataset ..."
        mkdir -p "$DATA_DIR"
        cd "$VENDOR_DIR"
        uvrun python scripts_build_index/decrypt_dataset.py \
            --output "$DATA_DIR/browsecomp_plus_decrypted.jsonl" \
            --generate-tsv "$VENDOR_DIR/topics-qrels/queries.tsv"
        cd "$PROJECT_ROOT"
        echo "Dataset decrypted to $DATA_DIR/browsecomp_plus_decrypted.jsonl"
    else
        echo "Dataset already decrypted"
    fi

    # Check Ollama is running
    if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "Ollama is running at $OLLAMA_URL"
    else
        echo "WARNING: Ollama not reachable at $OLLAMA_URL"
    fi

    echo "=== Setup complete ==="
}

cmd_download_index() {
    echo "=== Downloading pre-built BM25 index ==="
    cd "$VENDOR_DIR"
    # Only download BM25 (skip the large embedding indexes unless needed)
    uvrun huggingface-cli download Tevatron/browsecomp-plus-indexes \
        --repo-type=dataset --include="bm25/*" --local-dir ./indexes
    cd "$PROJECT_ROOT"
    echo "BM25 index downloaded to $VENDOR_DIR/indexes/bm25/"
}

cmd_build_lancedb() {
    echo "=== Building LanceDB index ==="
    uv sync --project "$SCRIPT_DIR" --extra lancedb
    uvrun python "$SCRIPT_DIR/lancedb_searcher.py" \
        --build-index \
        --db-path "$DATA_DIR/lancedb" \
        --ollama-url "$OLLAMA_URL" \
        --embedding-model "$GW_EMBED_MODEL"
}

cmd_build_bm25s() {
    local index_path="${1:-$DATA_DIR/bm25s-index}"

    echo "=== Building bm25s index ==="
    uv sync --project "$SCRIPT_DIR" --extra bm25s
    uvrun python "$SCRIPT_DIR/bm25s_searcher.py" \
        --build-index \
        --index-path "$index_path"
}

cmd_bm25s() {
    local index_path="${1:-$DATA_DIR/bm25s-index}"
    local output_dir="${2:-$VENDOR_DIR/runs/bm25s/ollama-$GW_MODEL}"

    echo "=== Running BrowseComp-Plus with bm25s + Ollama ($GW_MODEL) ==="
    echo "  Index: $index_path"
    echo "  Output: $output_dir"

    uv sync --project "$SCRIPT_DIR" --extra bm25s
    uvrun python "$SCRIPT_DIR/ollama_client.py" \
        --ollama-url "$OLLAMA_URL" \
        --model "$GW_MODEL" \
        --searcher-type bm25s \
        --index-path "$index_path" \
        --query "topics-qrels/queries.tsv" \
        --output-dir "$output_dir" \
        --k 5 \
        --max-turns 10
}

cmd_bm25() {
    local index_path="${1:-$DATA_DIR/indexes/browsecomp-plus-lucene}"
    local output_dir="${2:-$VENDOR_DIR/runs/bm25/ollama-$GW_MODEL}"

    echo "=== Running BrowseComp-Plus with BM25 (Pyserini) + Ollama ($GW_MODEL) ==="
    echo "  Index: $index_path"
    echo "  Output: $output_dir"

    uv sync --project "$SCRIPT_DIR" --extra bm25
    uvrun python "$SCRIPT_DIR/ollama_client.py" \
        --ollama-url "$OLLAMA_URL" \
        --model "$GW_MODEL" \
        --searcher-type bm25 \
        --index-path "$index_path" \
        --query "topics-qrels/queries.tsv" \
        --output-dir "$output_dir" \
        --k 5 \
        --max-turns 10
}

cmd_lancedb() {
    local db_path="${1:-$DATA_DIR/lancedb}"
    local output_dir="${2:-$VENDOR_DIR/runs/lancedb/ollama-$GW_MODEL}"

    echo "=== Running BrowseComp-Plus with LanceDB + Ollama ($GW_MODEL) ==="
    echo "  DB: $db_path"
    echo "  Output: $output_dir"

    uv sync --project "$SCRIPT_DIR" --extra lancedb
    uvrun python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$VENDOR_DIR/searcher')
sys.path.insert(0, '$VENDOR_DIR/search_agent')

from lancedb_searcher import LanceDBSearcher
from pathlib import Path
import argparse

args = argparse.Namespace(
    db_path='$db_path',
    table_name='browsecomp_docs',
    ollama_url='$OLLAMA_URL',
    embedding_model='$GW_EMBED_MODEL',
)

searcher = LanceDBSearcher(args)

from ollama_client import OllamaAgent, process_tsv

agent = OllamaAgent(
    ollama_url='$OLLAMA_URL',
    model='$GW_MODEL',
    searcher=searcher,
    k=5,
    max_turns=10,
)

tsv_path = Path('$VENDOR_DIR/topics-qrels/queries.tsv')
output_dir = Path('$output_dir')
process_tsv(agent, tsv_path, output_dir)
"
}

cmd_eval() {
    local run_dir="${1:?Usage: run.sh eval <run_dir>}"

    echo "=== Evaluating $run_dir ==="
    uv sync --project "$SCRIPT_DIR" --extra eval
    cd "$VENDOR_DIR"
    uvrun python scripts_evaluation/evaluate_run.py \
        --input_dir "$run_dir" \
        --ground_truth "$DATA_DIR/browsecomp_plus_decrypted.jsonl" \
        --eval_dir "$VENDOR_DIR/evals"
    cd "$PROJECT_ROOT"
}

cmd_quick() {
    local query="${1:-What is the capital of France?}"
    local output_dir="$VENDOR_DIR/runs/quick-test"

    echo "=== Quick test: single query ==="
    echo "  Query: $query"

    local index_path="$DATA_DIR/indexes/browsecomp-plus-lucene"
    if [ -d "$index_path" ]; then
        uv sync --project "$SCRIPT_DIR" --extra bm25
        uvrun python "$SCRIPT_DIR/ollama_client.py" \
            --ollama-url "$OLLAMA_URL" \
            --model "$GW_MODEL" \
            --searcher-type bm25 \
            --index-path "$index_path" \
            --query "$query" \
            --output-dir "$output_dir" \
            --max-turns 3
    else
        echo "No BM25 index found at $index_path"
        echo "Run './bench/browsecomp/run.sh setup' and './bench/browsecomp/run.sh download-index' first"
        exit 1
    fi
}

# --------------------------------------------------------------------------- #

case "${1:-help}" in
    setup)          cmd_setup ;;
    download-index) cmd_download_index ;;
    build-bm25s)    shift; cmd_build_bm25s "$@" ;;
    build-lancedb)  cmd_build_lancedb ;;
    bm25s)          shift; cmd_bm25s "$@" ;;
    bm25)           shift; cmd_bm25 "$@" ;;
    lancedb)        shift; cmd_lancedb "$@" ;;
    eval)           shift; cmd_eval "$@" ;;
    quick)          shift; cmd_quick "$@" ;;
    *)
        echo "Greatwheel BrowseComp-Plus Benchmark"
        echo ""
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  setup              Create venv, sync deps, decrypt dataset"
        echo "  download-index     Download pre-built BM25 Lucene index"
        echo "  build-bm25s [path] Build bm25s index (pure Python, no Java)"
        echo "  build-lancedb      Build LanceDB vector index (needs Ollama)"
        echo "  bm25s [idx] [out]  Run agent with bm25s retrieval"
        echo "  bm25 [idx] [out]   Run agent with Pyserini BM25 (needs Java 21)"
        echo "  lancedb [db] [out] Run agent with LanceDB retrieval"
        echo "  eval <run_dir>     Evaluate a run directory"
        echo "  quick [query]      Quick single-query test"
        echo ""
        echo "Environment:"
        echo "  OLLAMA_URL=$OLLAMA_URL"
        echo "  GW_MODEL=$GW_MODEL"
        echo "  GW_EMBED_MODEL=$GW_EMBED_MODEL"
        ;;
esac
