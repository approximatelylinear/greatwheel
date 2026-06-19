#!/usr/bin/env bash
#
# Spawn a fresh git worktree for a BrowseComp bench experiment.
#
# Usage:
#   scripts/bench-worktree-spawn.sh <slug> [base-branch]
#
# Creates ../greatwheel-exp-<slug>, branches bench/<slug> from base-branch
# (default: current HEAD), symlinks shared corpus + indexes from the main
# checkout, reserves a search-server port, prints an env block.
#
# Emits key=value lines on stdout for the caller to source. Progress goes
# to stderr so `eval "$(bench-worktree-spawn.sh foo)"` works.
#
set -euo pipefail

log() { echo "[spawn] $*" >&2; }
die() { echo "[spawn] ERROR: $*" >&2; exit 1; }

# --- args ------------------------------------------------------------------

[ $# -ge 1 ] || die "usage: $0 <slug> [base-branch]"
RAW_SLUG="$1"
BASE_BRANCH="${2:-HEAD}"

# sanitize: lowercase, [a-z0-9_-], no leading/trailing dash
SLUG="$(echo "$RAW_SLUG" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g; s/^-*//; s/-*$//')"
[ -n "$SLUG" ] || die "slug is empty after sanitization"

# --- locate main checkout --------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# If we were invoked from a worktree, resolve the actual main checkout via
# `git worktree list --porcelain` — first `worktree` entry is primary.
PRIMARY="$(git -C "$MAIN_ROOT" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
[ -n "$PRIMARY" ] && MAIN_ROOT="$PRIMARY"

PARENT="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="$PARENT/greatwheel-exp-$SLUG"
BRANCH="bench/$SLUG"

# --- port allocation -------------------------------------------------------

# Deterministic offset from slug: first 8 hex chars of sha1, mod 100.
HASH_HEX="$(echo -n "$SLUG" | sha1sum | cut -c1-8)"
PORT_OFFSET=$(( 16#$HASH_HEX % 100 ))
SEARCH_SERVER_PORT=$(( 8000 + PORT_OFFSET ))

if command -v ss >/dev/null 2>&1; then
    if ss -ltn "sport = :$SEARCH_SERVER_PORT" | tail -n +2 | grep -q .; then
        die "port $SEARCH_SERVER_PORT already in use (slug collision?); pick a different slug"
    fi
fi

# --- create worktree -------------------------------------------------------

if [ -e "$WORKTREE_PATH" ]; then
    die "worktree path already exists: $WORKTREE_PATH"
fi

if git -C "$MAIN_ROOT" show-ref --quiet "refs/heads/$BRANCH"; then
    log "branch $BRANCH exists — checking out into new worktree"
    git -C "$MAIN_ROOT" worktree add "$WORKTREE_PATH" "$BRANCH" >&2
else
    log "creating branch $BRANCH from $BASE_BRANCH"
    git -C "$MAIN_ROOT" worktree add -b "$BRANCH" "$WORKTREE_PATH" "$BASE_BRANCH" >&2
fi

# --- symlink shared resources ---------------------------------------------

link_shared() {
    local rel="$1"
    local src="$MAIN_ROOT/$rel"
    local dst="$WORKTREE_PATH/$rel"
    if [ ! -e "$src" ]; then
        log "skip (not present in main): $rel"
        return
    fi
    # worktree checkout may have created an empty dir; nuke only if empty.
    if [ -d "$dst" ] && [ -z "$(ls -A "$dst" 2>/dev/null || true)" ]; then
        rmdir "$dst"
    fi
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        log "skip (already present in worktree): $rel"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
    log "linked $rel"
}

link_shared "vendor/BrowseComp-Plus"
link_shared "data/qwen3-embed"
link_shared "data/kb-bc-lancedb"
link_shared "data/kb-bc-tantivy"
link_shared "data/colbert-lance"

# --- per-worktree results dir ---------------------------------------------

mkdir -p "$WORKTREE_PATH/results"

# --- emit env block --------------------------------------------------------

cat <<EOF
GW_WORKTREE=$WORKTREE_PATH
GW_SLUG=$SLUG
GW_BRANCH=$BRANCH
QWEN3_EMBED_URL=http://localhost:8003
COLBERT_URL=http://localhost:8002
SEARCH_SERVER_PORT=$SEARCH_SERVER_PORT
EOF

log "ready: cd $WORKTREE_PATH"
