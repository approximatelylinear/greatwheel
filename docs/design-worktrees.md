# Design: Parallel Worktree Support

**Status:** Sketch
**Date:** 2026-04-17

## Motivation

Working on greatwheel from a single checkout serializes everything: one branch
at a time, one `cargo check` at a time, and one `target/` warming cycle at a
time. Git worktrees would let us run multiple branches in parallel (e.g. a
long refactor alongside a quick bench experiment), but today several
assumptions in the config and tooling collide when two worktrees try to run
simultaneously:

- `[database].url` points at a single Postgres DB (`greatwheel`)
- `[server].port` is hardcoded to 8090
- `data/*` dirs are cwd-relative (good — already isolated) but the migrations
  runner implicitly targets the shared DB
- LLM backends (Ollama, sglang) are singletons bound to well-known ports
  and, on sglang, to the single GPU

We want a runtime process where spinning up a worktree is a one-command
operation that yields a fully isolated server without re-provisioning the
heavyweight shared services.

## Goals

- Two or more worktrees running `cargo run --bin greatwheel` concurrently
  without port, DB, or data-dir collisions
- Zero duplication of LLM weights, HF cache, or model servers
- A predictable naming scheme so the user can tell at a glance which worktree
  owns which DB / port
- Cheap teardown — removing a worktree leaves no orphaned state that blocks
  reuse of the same branch name later

## Non-goals

- Running multiple sglang instances (GPU-bound, not worth it)
- Sharing `target/` via `CARGO_TARGET_DIR` (known lock-contention footgun;
  disk is cheap, keep per-worktree `target/`)
- Isolating Ollama model downloads per worktree
- Production multi-tenancy — this is a dev-ergonomics change only

## Shared-resource inventory

| Resource | Today | Per-worktree or shared? | Why |
|---|---|---|---|
| Postgres server | 1 container, port 5432 | **Shared server, isolated DB** | Schema-per-worktree is cheap; running N Postgres containers is not |
| Postgres database | `greatwheel` | **Isolated** (`greatwheel_<wt>`) | Migrations, session tables, traces all diverge per branch |
| Ollama | Port 11434 | **Shared** | Model weights are huge; API is stateless across callers |
| sglang | Port 30000, 1 GPU | **Shared** | GPU is singular; RadixAttention cache is prefix-scoped so cross-worktree contention is mild |
| rl-play proxy | Port 8000 | **Shared** | Stateless router in front of Ollama |
| HF cache | `hf-cache` volume | **Shared** | Downloaded weights are identical across branches |
| `gw-server` HTTP | Port 8090 | **Isolated** (port offset) | Each worktree needs its own listener |
| `data/lancedb`, `data/tantivy` | cwd-relative | **Already isolated** | Worktree cwd differs; no change needed |
| `data/kb-*` | cwd-relative | **Shared by symlink, opt-out per worktree** | Re-ingestion is expensive; default to a shared KB, break the symlink when a worktree changes KB schema |
| `target/` | Per-worktree default | **Already isolated** | Leave as is |
| Tracing export (OTel/console) | `postgres_export = false` in dev | **Shared if enabled** | Low volume; not worth isolating |

## Proposed design

### 1. Worktree identity

Derive a stable short ID from the worktree path:

```
WORKTREE_ID = basename($PWD)        # e.g. "gw-refactor", "gw-browsecomp"
WORKTREE_SLUG = sanitize($WORKTREE_ID)   # lowercase, [a-z0-9_]
```

The main checkout keeps ID `greatwheel` (no suffix in DB names, default port)
so existing workflows are undisturbed. Every other worktree gets a suffix.

### 2. Port allocation

Deterministic offset from the slug, not sequential, so ports are stable
across restarts:

```
PORT_OFFSET = hash(WORKTREE_SLUG) % 100
SERVER_PORT = 8090 + PORT_OFFSET
```

Collisions at 1/100 are acceptable for ≤5 concurrent worktrees; a collision
check at boot turns a crash into a clear error message.

### 3. Config layering

Introduce `config/greatwheel.local.toml` as an overlay loaded *after*
`config/greatwheel.toml`. The overlay is `.gitignore`d. A bootstrap script
(next section) writes it per worktree. The server's config loader gains a
simple merge step — overlay wins.

This avoids a full templating engine and keeps the committed config as the
canonical baseline.

### 4. Bootstrap script: `scripts/worktree-init.sh`

One-shot per worktree:

1. Compute `WORKTREE_SLUG` and `SERVER_PORT`
2. `psql -c "CREATE DATABASE greatwheel_<slug>"` (idempotent via `IF NOT EXISTS` pattern)
3. Run `sqlx migrate run` against the new DB
4. Symlink shared KB data if not already present:

   ```
   ln -sfn <main-checkout>/data/kb-lancedb data/kb-lancedb
   ln -sfn <main-checkout>/data/kb-tantivy data/kb-tantivy
   ```

   The main checkout path is resolved via `git worktree list --porcelain`
   (the first entry is the primary). To opt out — e.g. when a worktree
   mutates KB schema — `rm data/kb-lancedb && gw-kb ingest ...` produces a
   local copy and subsequent runs use it.
5. Write `config/greatwheel.local.toml`:

   ```toml
   [server]
   port = <SERVER_PORT>

   [database]
   url = "postgres://gw:gw@localhost:5432/greatwheel_<slug>"
   ```

6. Print the server URL, DB name, and whether KB is symlinked or local

### 5. Runtime script: `scripts/worktree-run.sh`

Thin wrapper — just `cargo run --bin greatwheel -- --config config/greatwheel.toml`.
The overlay is picked up automatically. Kept as a separate script so we can
add env-var plumbing later (e.g. `GW_WORKTREE=<slug>` for log prefixes).

### 6. Cleanup: `scripts/worktree-cleanup.sh`

Takes a slug (or `--stale`, which enumerates `greatwheel_*` DBs and drops any
whose matching worktree no longer exists). Drops the DB, removes the overlay,
optionally clears `data/` for that worktree. Must check `data/kb-*` before
deleting — a symlink gets `unlink`ed, a real dir only gets removed with
`--force-kb` to avoid nuking a local re-ingestion the user cared about.

### 7. Docker Compose changes

Split `docker-compose.yml` into two files:

- `docker-compose.shared.yml` — `postgres`, `ollama`, `sglang`, `rl-play`.
  Started once, stays up across worktrees.
- `docker-compose.app.yml` — `greatwheel` only, parameterized by
  `WORKTREE_SLUG` / `SERVER_PORT` env vars.

In practice most dev work won't use the app container at all (we run
`cargo run` natively), so the main win here is documenting the split.

## Config loader changes

In `crates/gw-server/src/main.rs`, replace the single-file load with a
two-file merge. Simplest path: load base TOML → `toml::Value`, load overlay
if present → deep-merge → deserialize into `Config`. Adds ~20 lines, no new
dependency.

## Open questions

- **KB write safety under symlink.** Shared-by-default means a worktree that
  accidentally runs `gw-kb ingest` or `organize` will mutate the main
  checkout's KB. Mitigations to pick from: (a) make KB-write commands refuse
  to run when `data/kb-*` is a symlink unless `--allow-shared-write` is
  passed; (b) flip the symlink to read-only on the filesystem; (c) rely on
  convention and fix in review. (a) is the most defensive and probably
  worth the ~10 lines in `gw-kb`.
- **Migrations on branch switch.** If worktree A runs migration 013 and
  worktree B is on a branch without it, switching back to A's DB from B
  doesn't break anything, but running `sqlx migrate run` from B would see
  an "ahead" DB. `sqlx` handles this gracefully today; worth a test.
- **Bench artifacts.** `bench/browsecomp/gepa_runs/` and `runs/` are cwd-
  relative so already isolated, but they can be large. Add to the cleanup
  script.
- **Trace DB.** If `postgres_export = true` is ever enabled in dev, traces
  land in the per-worktree DB — probably fine, but worth naming.

## Out of scope / future

- A TUI / CLI (`gw worktree new <branch>`) wrapping the scripts
- Per-worktree agent hot-reload namespaces (today `agents/` is shared via
  repo — each worktree already has its own copy, so no issue)
- Sharing `target/` via sccache (separate initiative)

## Rollout

1. Land the config overlay loader (small, standalone PR)
2. Add the three scripts with a short `docs/worktrees.md` usage note
3. Dogfood on one secondary worktree for a week before documenting in README
