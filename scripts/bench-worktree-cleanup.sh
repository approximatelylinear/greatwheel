#!/usr/bin/env bash
#
# Remove a bench worktree spawned by bench-worktree-spawn.sh.
#
# Usage:
#   scripts/bench-worktree-cleanup.sh <slug> [--force]
#   scripts/bench-worktree-cleanup.sh --stale [--force]
#
# Without --force, refuses to remove a worktree with uncommitted changes.
# --stale enumerates every worktree at ../greatwheel-exp-* whose branch is
# fully merged into origin/main (or whose branch no longer exists).
#
set -euo pipefail

log() { echo "[cleanup] $*" >&2; }
die() { echo "[cleanup] ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMARY="$(git -C "$MAIN_ROOT" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
[ -n "$PRIMARY" ] && MAIN_ROOT="$PRIMARY"
PARENT="$(dirname "$MAIN_ROOT")"

FORCE=0
MODE=""
SLUG=""

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --stale) MODE="stale" ;;
        --*) die "unknown flag: $arg" ;;
        *)
            [ -z "$SLUG" ] || die "multiple slugs given"
            SLUG="$arg"
            MODE="${MODE:-one}"
            ;;
    esac
done

[ -n "$MODE" ] || die "usage: $0 <slug> [--force] | --stale [--force]"

remove_one() {
    local slug="$1"
    local path="$PARENT/greatwheel-exp-$slug"
    local branch="bench/$slug"

    if [ ! -d "$path" ]; then
        log "skip: $path does not exist"
        return
    fi

    local flags=""
    [ "$FORCE" -eq 1 ] && flags="--force"

    log "removing worktree $path"
    # shellcheck disable=SC2086
    git -C "$MAIN_ROOT" worktree remove $flags "$path" || {
        die "worktree has uncommitted changes; pass --force to discard"
    }

    if git -C "$MAIN_ROOT" show-ref --quiet "refs/heads/$branch"; then
        if git -C "$MAIN_ROOT" branch --merged origin/main 2>/dev/null | grep -qxF "  $branch"; then
            git -C "$MAIN_ROOT" branch -d "$branch" >&2 && log "deleted merged branch $branch"
        else
            log "kept branch $branch (not merged into origin/main)"
        fi
    fi
}

case "$MODE" in
    one)
        remove_one "$SLUG"
        ;;
    stale)
        git -C "$MAIN_ROOT" worktree prune
        for path in "$PARENT"/greatwheel-exp-*; do
            [ -d "$path" ] || continue
            slug="${path##*/greatwheel-exp-}"
            branch="bench/$slug"
            if ! git -C "$MAIN_ROOT" show-ref --quiet "refs/heads/$branch"; then
                log "$slug: branch gone, removing worktree"
                remove_one "$slug"
                continue
            fi
            if git -C "$MAIN_ROOT" branch --merged origin/main 2>/dev/null | grep -qxF "  $branch"; then
                log "$slug: merged, removing"
                remove_one "$slug"
            else
                log "$slug: still active, skipping"
            fi
        done
        ;;
esac
