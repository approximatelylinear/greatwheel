#  Design: Parallel Bench Worktrees

**Status:** Sketch
**Date:** 2026-04-24

## Motivation

We want to run N retrieval experiments in parallel — each a subagent with
its own git worktree, its own config, and its own results file — so a
supervisor agent can fan out across the top-k untested hypotheses from
`bench/browsecomp/EXPERIMENTS.md`, wait for all to finish, then synthesize
cross-experiment rankings.

This is a narrower problem than `docs/design-worktrees.md`, which targets
running `gw-server` (Postgres, port 8090, migrations) in parallel. For
bench experiments none of that applies — the collisions are different and
the resources that need sharing are different.

## Goals

- A supervisor agent can spawn K subagents, each in a fresh worktree,
  running a BrowseComp config end-to-end, writing results to a stable
  path the supervisor can read back.
- No duplication of the corpus, the qwen3 index, or model weights —
  these are ~GB-scale and identical across experiments that don't mutate
  indexing.
- The GPU-bound qwen3 embed server stays a singleton; subagents share it.
- Cheap teardown: `git worktree remove` plus one symlink cleanup is
  enough to leave no orphaned state.

## Non-goals

- Running the Rust `gw-server` in parallel (see `design-worktrees.md`
  when that's needed).
- Parallelizing experiments that rebuild indexes with *different*
  embedding models — those serialize on the single GPU during index
  build, then can fan out for eval.
- Postgres / sqlx isolation — bench doesn't touch either.

## Resource inventory

| Resource | Handling | Why |
|---|---|---|
| `vendor/BrowseComp-Plus/` (corpus, ~GB) | **Symlink to main** | Read-only, identical across experiments |
| `data/qwen3-embed/` (LanceDB index) | **Symlink to main**, opt-out per worktree | Rebuilding is minutes on GPU; most experiments vary query-time params only |
| `data/kb-bc-lancedb`, `data/kb-bc-tantivy` | **Symlink to main**, opt-out | Same reasoning |
| `qwen3_embed_server.py` (port 8003) | **Shared singleton** | Single GPU. Subagents connect to the main checkout's instance |
| `colbert_server.py` (port 8002) | **Shared singleton** | Same |
| `search_server_qwen3.py` (port 8000) | **Per-worktree, port-offset** | Stateless wrapper, config-dependent |
| Ollama / rl-play | **Shared** | Stateless across callers |
| `bench/browsecomp/.venv` | **Per-worktree** | Fast rebuild via uv cache; avoids interpreter-path breakage on symlink |
| `bench/browsecomp/gepa_runs/`, `runs/`, `results/` | **Per-worktree** (cwd-relative already) | No change needed |
| `target/` | **Per-worktree** (default) | No change needed |

## Worktree layout

```
~/Code/
├── greatwheel/                          # main checkout
│   ├── vendor/BrowseComp-Plus/          # real, ~GB
│   ├── data/qwen3-embed/                # real index
│   └── bench/browsecomp/.venv/          # real venv
└── greatwheel-exp-<slug>/               # spawned worktree
    ├── vendor/BrowseComp-Plus -> ../../greatwheel/vendor/BrowseComp-Plus
    ├── data/qwen3-embed -> ../../greatwheel/data/qwen3-embed
    └── bench/browsecomp/.venv/          # own venv (uv sync on first run)
```

## Slug & port allocation

```
SLUG = sanitize(<experiment-name>)           # lowercase, [a-z0-9_-]
PORT_OFFSET = hash(SLUG) % 100
SEARCH_SERVER_PORT = 8000 + PORT_OFFSET
```

Deterministic from the slug so restarts are stable. At boot the spawn
script probes the port and errors clearly on collision rather than
letting the server crash late.

## Shared services contract

Before fanning out, the supervisor ensures the shared singletons are up
in the **main checkout**:

- `qwen3_embed_server.py --port 8003`
- `colbert_server.py --port 8002` (if any experiment uses ColBERT)

Subagents are told (via env) `QWEN3_EMBED_URL=http://localhost:8003` and
MUST NOT spawn their own. If a subagent's hypothesis requires a
*different* embedding model, it's a serial experiment — the supervisor
runs it outside the parallel batch and restarts the singleton between.

## Results contract

Each subagent writes `results/<slug>.md` (cwd-relative, so per-worktree)
with a required YAML frontmatter block:

```markdown
---
slug: <slug>
hypothesis: "one-line description"
config: configs/<slug>.toml
baseline_commit: <sha>
scores:
  exact_match: 0.42
  fuzzy_match: 0.51
  r_at_200: 24
  r_at_200_bm25: 18
  r_at_200_colbert: 25
runtime_secs: 1240
---

<free-form notes, logs, failure modes>
```

The supervisor aggregates by reading each worktree's
`results/<slug>.md`, parsing the frontmatter. No DB, no JSON sidecar —
one file per experiment, diffable and human-readable.

## Spawn script: `scripts/bench-worktree-spawn.sh`

```
bench-worktree-spawn.sh <slug> [base-branch]
```

1. Compute `SLUG` and `SEARCH_SERVER_PORT`.
2. Pick worktree path: `<main-parent>/greatwheel-exp-<slug>`.
3. `git worktree add -b bench/<slug> <path> <base-branch|HEAD>`.
4. Symlink shared resources from the main checkout:
   - `vendor/BrowseComp-Plus`
   - `data/qwen3-embed`
   - `data/kb-bc-lancedb`
   - `data/kb-bc-tantivy`
5. `mkdir -p results/`.
6. Probe `SEARCH_SERVER_PORT`; error on collision.
7. Print an env block the supervisor/subagent sources:

   ```
   GW_WORKTREE=<path>
   GW_SLUG=<slug>
   QWEN3_EMBED_URL=http://localhost:8003
   SEARCH_SERVER_PORT=<port>
   ```

## Cleanup: `scripts/bench-worktree-cleanup.sh`

```
bench-worktree-cleanup.sh <slug>        # one
bench-worktree-cleanup.sh --stale       # every worktree whose results/<slug>.md is merged or >14d old
```

For each target: `git worktree remove` (refuses if dirty — caller passes
`--force` to override), then delete the branch if merged.

Symlinks are removed implicitly by `git worktree remove`. Shared
resources in main are never touched.

## Open questions

- **Opt-out from symlinked indexes.** If an experiment rebuilds
  `data/qwen3-embed/` it must first `rm data/qwen3-embed && mkdir
  data/qwen3-embed` to break the symlink. Worth a pre-flight check in
  `build_qwen3_index.py`: refuse to write if the path is a symlink
  unless `--allow-shared-write` is passed. Same pattern as the KB-write
  mitigation in `design-worktrees.md` §Open questions.
- **Venv duplication cost.** Each worktree gets its own `.venv`. On a
  warm uv cache this is <30s; on a cold cache closer to 2min. Cheap
  enough to not bother symlinking, and symlinked venvs break
  interpreter paths.
- **Supervisor synthesis atomicity.** If a subagent crashes mid-run, its
  `results/<slug>.md` won't exist. Supervisor treats missing files as
  failures, not zeros — synthesis skips them and flags in the writeup.

## Rollout

1. Land `scripts/bench-worktree-spawn.sh` + `scripts/bench-worktree-cleanup.sh`.
2. Adopt the `results/<slug>.md` frontmatter schema in one hand-driven
   experiment to confirm it's writable and diffable.
3. Write the supervisor prompt (lives outside this doc; probably a
   slash command or skill) that: reads `EXPERIMENTS.md`, picks top-K,
   spawns K subagents with the Task tool, each `cd $GW_WORKTREE && uv
   run ...`, waits, reads results, updates rankings.
4. Add a pre-flight check in `build_qwen3_index.py` refusing to write
   through a symlink without `--allow-shared-write`.
