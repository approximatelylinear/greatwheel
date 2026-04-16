# Design: Dead Code Cleanup & Code Quality

**Status:** All phases complete
**Date:** 2026-04-16
**Motivation:** Audit revealed orphaned crates, duplicate type definitions, copy-pasted
helpers, superseded Python scripts, and 57 compiler/clippy warnings. Cleaning this up
reduces cognitive load, speeds compilation, and removes traps for future contributors.

---

## 0. Scope

This document covers dead code removal and code quality improvements identified by a
full-codebase audit. Phases 1–3 have been implemented; phases 4–6 remain open.

### What was done

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Delete orphaned crates + fix compiler errors | **Done** |
| 2 | Mechanical enforcement (rustfmt, clippy, workspace lints) | **Done** |
| 3 | Code simplification across all crates | **Done** |
| 4 | Unify duplicate message types across crates | **Done** |
| 5 | Remove superseded Python scripts | **Done** |
| 6 | Housekeeping (gitignore, old runs) | **Done** |

### Results

- **Warnings**: 57 → 4 (93% reduction). Remaining 4 are "too many arguments" in the
  benchmark harness — acceptable for a leaf binary.
- **Tests**: 84 pass, 0 failures.
- **Dead crates removed**: 3 (`gw-bus`, `gw-channels`, `gw-scheduler`)
- **Duplicate types unified**: `LlmMessage` and `LlmResponse` canonical in `gw-core`
- **Dead Python scripts removed**: 8 (`retrieval_benchmark.py`, `build_colbert_index.py`,
  `rerank_server.py`, `test_colbert_passage.py`, `lancedb_searcher.py`,
  `build_voyager_index.py`, `build_usearch_index.py`, `voyager_searcher.py`)

---

## 1. Delete orphaned crates (done)

Three crates under `crates/` had zero consumers.

| Crate | Why it was dead |
|-------|-----------------|
| `gw-bus/` | Traits migrated to `gw-core/src/agent_bus.rs`; not in workspace members |
| `gw-channels/` | Traits migrated to `gw-core/src/channel.rs`; not in workspace members |
| `gw-scheduler/` | Empty stub with TODO comments; never imported in any `.rs` file |

**Changes:**
- Deleted `crates/gw-bus`, `crates/gw-channels`, `crates/gw-scheduler`
- Removed from root `Cargo.toml` workspace members and dependencies
- Removed `gw-scheduler` from `crates/gw-server/Cargo.toml`
- Updated `CLAUDE.md` crate layout

---

## 2. Mechanical enforcement (done)

### Tooling setup

- **`rustfmt.toml`** created — `cargo fmt` enforces consistent formatting
- **Workspace lints** in root `Cargo.toml`:
  ```toml
  [workspace.lints.rust]
  unused_imports = "deny"
  unused_variables = "deny"
  dead_code = "deny"

  [workspace.lints.clippy]
  redundant_closure = "warn"
  manual_split_once = "warn"
  needless_pass_by_value = "warn"
  large_enum_variant = "warn"
  let_unit_value = "warn"
  ```
- **`[lints] workspace = true`** added to all 11 crate `Cargo.toml` files

### Dead code removed

| Item | File |
|------|------|
| 12 unused imports | 6 files across gw-bench, gw-server, conv-loop-bench |
| `BrowseCompBridge::log_timing_summary()` | `gw-bench/src/main.rs` |
| `QUERY_TIMEOUT_SECS` constant | `gw-bench/src/main.rs` |
| Unused struct fields `max_iterations`, `bench_config` | `gw-bench/src/conv_loop_runner.rs` |
| `MockCall` struct + `MockBridge` fields | `bench/conv-loop/src/mock.rs` (simplified to unit struct) |
| `let result =` on unit value | `gw-server/src/session_api.rs` |

### Clippy fixes applied

| Category | Count | Examples |
|----------|-------|---------|
| `clone` → `from_ref` | 3 | Single-element slices in gw-memory, gw-bench |
| Doc list overindentation | 9 | gw-memory, gw-kb |
| Field assignment outside initializer | 1 | gw-trace `postgres_layer.rs` |
| `&mut Vec` → `&mut [_]` | 1 | gw-kb `clean.rs` |
| Literal in format string | 3 | gw-kb binary |
| Redundant import | 1 | conv-loop-bench |

---

## 3. Code simplification (done)

### Duplicate code extracted

| Duplication | Where | Fix |
|-------------|-------|-----|
| `strip_code_fences()` (2 copies) + JSON extraction (3 copies) | gw-kb: organize, classify, merge | New `gw-kb/src/llm_parse.rs` with `strip_code_fences()` and `extract_json<T>()` |
| Embedding vector parsing (3 copies) | gw-llm `embed()` | Extracted `parse_embedding()` helper |
| REPL replay from code executions (2 copies) | gw-loop `session.rs` | Deduplicated with labeled block + early returns |
| Duplicate constructors `new()` / `with_pg()` | gw-loop `SessionManager` | Unified to single `new()` with `Option<PgPool>` |

### Catch-all matches → exhaustive

Replaced `_ => {}` with explicit variant listings so the compiler catches
unhandled cases when new variants are added:

| File | Enum matched |
|------|-------------|
| `gw-loop/src/context.rs` | `EntryType` |
| `gw-loop/src/conversation.rs` (2 sites) | `LoopEvent`, `EntryType` |
| `gw-bench/src/conv_loop_runner.rs` | `EntryType` |

### Complex types → type aliases

| Alias | Crate | Replaces |
|-------|-------|----------|
| `AnswerValidator` | gw-loop | `Box<dyn Fn(&str) -> bool + Send>` |
| `EntryRow` | gw-loop | 6-element sqlx tuple |
| `HostFnHandlerAsync` (existing, now used) | gw-core | Complex `Arc<dyn Fn(...) -> BoxFuture<...>>` |
| `MemoryRow` | gw-memory | 7-element sqlx tuple |
| `UpsertParams` struct | gw-memory | 8-argument function signature |
| `TopicStateRow`, `TopicRow`, `ChunkRow` | gw-kb/topics | sqlx tuples |
| `SourceListRow` | gw-kb/source | 8-element sqlx tuple |
| `NewSourceRow` | gw-kb/digest | 7-element sqlx tuple |

### Pass-by-value → reference

| Function | Change |
|----------|--------|
| `ConversationLoop::inject_steering` | `String` → `&str` |
| `gw-kb/src/index.rs::lance_err` | `lancedb::Error` → `&lancedb::Error` |

---

## 4. Unify message types across crates (done)

Canonical `LlmMessage` and `LlmResponse` structs now live in `gw-core/src/lib.rs`.

**Changes:**
- Defined `LlmMessage { role, content }` and `LlmResponse { content, model, input_tokens, output_tokens }` in `gw-core`
- `gw-llm`: removed local `Message` and `CompletionResponse` structs; re-exports canonical types + type aliases for backward compat
- `gw-loop/src/context.rs`: removed local `LlmMessage`; imports from `gw-core`
- `gw-loop/src/llm.rs`: removed local `LlmResponse`; `OllamaLlmClient` adapter no longer needs field-by-field conversion
- `gw-core/src/plugin.rs`: removed `LlmMessageData`; `EventData::Messages` uses `LlmMessage` directly
- `gw-bench/src/main.rs`: updated error-path struct literals to include `model: None`

---

## 5. Remove superseded Python scripts (done)

| Dead file | Replacement | Reason |
|-----------|-------------|--------|
| `retrieval_benchmark.py` | `retrieval_benchmark_v2.py` | v2 uses pluggable `searchers/` architecture |
| `build_colbert_index.py` | `build_colbert_passage_index.py` | Passage-level version is newer, has resume logic |
| `test_colbert_passage.py` | — | One-off manual test, not in any test suite |
| `rerank_server.py` | `rerank_server_blobs.py` | Blob version is 450x faster |
| `lancedb_searcher.py` | `searchers/lancedb_mv_searcher.py` | Old `BaseSearcher` interface superseded |
| `build_voyager_index.py` | — | Voyager backend abandoned (108 GB index, no PQ) |
| `build_usearch_index.py` | — | USearch backend not in active benchmarks |
| `voyager_searcher.py` | — | Voyager backend abandoned |

**Changes:**
- Extracted `split()` and `title()` helpers to new `text_utils.py`
- Updated `build_passage_blob_store.py` to import from `text_utils`
- Removed LanceDB code paths from `search_server.py`, `ollama_client.py`, `run.sh`
- Updated stale comments in `gw-memory` referencing deleted files

---

## 6. Housekeeping (done)

| Item | Action |
|------|--------|
| `docs/*.aux`, `docs/*.log`, `docs/*.out` | Added to `.gitignore` |
| `runs/` pre-April 8 directories | Left in place (untracked, ~2.5 GB) |
| `docs/design-episodes.md` | Left as-is — unimplemented design, no code exists |
