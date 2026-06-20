//! Integration tests for `BenchStore` against the fixture run dirs in
//! `tests/data/`. Two fake experiments × 3 queries each, with one
//! exact-match, one wrong-answer, one fuzzy-only case per run.
//!
//! See `tests/data/runs/{exp-a,exp-b}/` and `tests/data/gold.jsonl`
//! for the inputs.

use std::path::PathBuf;

use gw_bench_ui::types::{MatrixCell, Winner};
use gw_bench_ui::{BenchPaths, BenchStore};

fn store() -> BenchStore {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let paths = BenchPaths::new(
        root.join("runs"),
        root.join("configs"),
        root.join("gold.jsonl"),
    );
    BenchStore::new(paths)
}

#[test]
fn list_experiments_finds_both() {
    let s = store();
    let rows = s.list_experiments().unwrap();
    let slugs: Vec<&str> = rows.iter().map(|r| r.slug.as_str()).collect();
    assert!(slugs.contains(&"exp-a"), "missing exp-a in {slugs:?}");
    assert!(slugs.contains(&"exp-b"), "missing exp-b in {slugs:?}");
    assert_eq!(rows.len(), 2);
}

#[test]
fn load_run_aggregates_exp_a() {
    let s = store();
    let detail = s.load_run("exp-a").unwrap();
    assert_eq!(detail.aggregate.n_queries, 3);
    // q1 exact, q2 wrong, q3 fuzzy-only.
    assert_eq!(detail.aggregate.exact, 1);
    assert_eq!(detail.aggregate.fuzzy, 2);
    let q1 = detail
        .queries
        .iter()
        .find(|q| q.query_id == "1001")
        .unwrap();
    assert!(q1.exact);
    assert!(q1.fuzzy);
    let q2 = detail
        .queries
        .iter()
        .find(|q| q.query_id == "1002")
        .unwrap();
    assert!(!q2.exact);
    assert!(!q2.fuzzy);
    let q3 = detail
        .queries
        .iter()
        .find(|q| q.query_id == "1003")
        .unwrap();
    assert!(!q3.exact, "Camberra ≠ Canberra exactly");
    assert!(q3.fuzzy, "edit distance 1/8 < 0.3 → fuzzy");
}

#[test]
fn load_run_aggregates_exp_b() {
    let s = store();
    let detail = s.load_run("exp-b").unwrap();
    assert_eq!(detail.aggregate.exact, 2);
    assert_eq!(detail.aggregate.fuzzy, 2);
    // p50 over [60_000, 90_000, 150_000] → 90_000.
    assert_eq!(detail.aggregate.p50_ms, Some(90_000));
    assert_eq!(detail.aggregate.total_input_tokens, 3000 + 6000 + 9000);
}

#[test]
fn load_query_returns_full_detail() {
    let s = store();
    let q = s.load_query("exp-a", "1001").unwrap();
    assert_eq!(q.gold_answer, "Tagalog");
    assert_eq!(q.agent_answer, "Tagalog");
    assert!(q.exact);
    assert_eq!(q.iterations, 4);
    assert!(q.question.contains("Philippines"));
}

#[test]
fn load_query_missing_errors() {
    let s = store();
    let err = s.load_query("exp-a", "9999").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("9999"),
        "expected NotFound for 9999, got {msg}"
    );
}

#[test]
fn load_config_finds_existing() {
    let s = store();
    let info = s.load_config("exp-a").unwrap();
    assert!(info.toml_path.is_some(), "exp-a.toml should exist");
    assert!(info.contents.unwrap().contains("rerank_candidate_pool"));
}

#[test]
fn load_config_missing_returns_none() {
    let s = store();
    let info = s.load_config("exp-b").unwrap();
    assert!(info.toml_path.is_none());
    assert!(info.contents.is_none());
}

#[test]
fn gold_lookup() {
    let s = store();
    let entry = s.gold("1001").unwrap().unwrap();
    assert_eq!(entry.answer, "Tagalog");
    assert!(s.gold("nonexistent").unwrap().is_none());
}

#[test]
fn compare_classifies_winners() {
    let s = store();
    let cmp = s.compare("exp-a", "exp-b").unwrap();
    let row = |qid: &str| {
        cmp.per_query
            .iter()
            .find(|r| r.query_id == qid)
            .unwrap_or_else(|| panic!("missing {qid}"))
    };
    // q1: both exact → BothCorrect.
    assert!(matches!(row("1001").winner, Winner::BothCorrect));
    // q2: a wrong, b exact → B.
    assert!(matches!(row("1002").winner, Winner::B));
    // q3: a fuzzy-only, b wrong → A wins on fuzzy.
    assert!(matches!(row("1003").winner, Winner::A));
    assert_eq!(cmp.deltas.exact_a, 1);
    assert_eq!(cmp.deltas.exact_b, 2);
}

#[test]
fn read_annotations_picks_up_fixture() {
    let s = store();
    let ann = s.read_annotations("exp-a").unwrap();
    assert_eq!(ann.notes.len(), 1);
    assert!(ann.notes[0].text.contains("Baseline"));
    assert_eq!(ann.tags, vec!["baseline", "qwen3.5:9b"]);
}

#[test]
fn read_annotations_missing_file_returns_default() {
    let s = store();
    let ann = s.read_annotations("exp-b").unwrap();
    assert!(ann.notes.is_empty());
    assert!(ann.tags.is_empty());
}

#[test]
fn load_run_merges_annotations() {
    let s = store();
    let detail = s.load_run("exp-a").unwrap();
    assert_eq!(detail.notes.len(), 1);
    assert_eq!(detail.tags, vec!["baseline", "qwen3.5:9b"]);
    let exp_a = s
        .list_experiments()
        .unwrap()
        .into_iter()
        .find(|r| r.slug == "exp-a")
        .unwrap();
    assert_eq!(exp_a.tags, vec!["baseline", "qwen3.5:9b"]);
}

#[test]
fn write_annotation_appends_and_unions_tags() {
    // Stage a tempdir with one empty run dir + an empty gold file so
    // BenchStore can resolve the slug. Writes go into the tempdir, not
    // the fixture tree.
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("runs");
    std::fs::create_dir_all(runs.join("expW")).unwrap();
    let gold = tmp.path().join("gold.jsonl");
    std::fs::write(&gold, "").unwrap();
    let configs = tmp.path().join("configs");
    std::fs::create_dir(&configs).unwrap();
    let s = BenchStore::new(BenchPaths::new(runs.clone(), configs, gold));

    s.write_annotation("expW", "first note", vec!["a".into(), "b".into()])
        .unwrap();
    s.write_annotation("expW", "second note", vec!["b".into(), "c".into()])
        .unwrap();

    let ann = s.read_annotations("expW").unwrap();
    assert_eq!(ann.notes.len(), 2);
    assert_eq!(ann.notes[0].text, "first note");
    assert_eq!(ann.notes[1].text, "second note");
    // Tags union, preserving insertion order, deduping `b`.
    assert_eq!(ann.tags, vec!["a", "b", "c"]);

    // Sidecar lives at the expected path.
    assert!(runs.join("expW/notes.json").exists());
}

#[test]
fn write_annotation_without_text_just_adds_tags() {
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("runs");
    std::fs::create_dir_all(runs.join("expT")).unwrap();
    let s = BenchStore::new(BenchPaths::new(
        runs,
        tmp.path().join("configs"),
        tmp.path().join("gold.jsonl"),
    ));
    s.write_annotation("expT", "", vec!["only-tag".into()])
        .unwrap();
    let ann = s.read_annotations("expT").unwrap();
    assert!(ann.notes.is_empty());
    assert_eq!(ann.tags, vec!["only-tag"]);
}

#[test]
fn difficulty_matrix_classifies_cells() {
    let s = store();
    let m = s
        .difficulty_matrix(&["exp-a".into(), "exp-b".into()])
        .unwrap();
    assert_eq!(m.runs, vec!["exp-a", "exp-b"]);
    assert_eq!(m.queries, vec!["1001", "1002", "1003"]);
    assert_eq!(m.cells.len(), 3);

    let cell = |q_idx: usize, r_idx: usize| m.cells[q_idx][r_idx];
    // 1001: both exact.
    assert!(matches!(cell(0, 0), MatrixCell::Exact));
    assert!(matches!(cell(0, 1), MatrixCell::Exact));
    // 1002: a wrong, b exact.
    assert!(matches!(cell(1, 0), MatrixCell::Wrong));
    assert!(matches!(cell(1, 1), MatrixCell::Exact));
    // 1003: a fuzzy, b wrong.
    assert!(matches!(cell(2, 0), MatrixCell::Fuzzy));
    assert!(matches!(cell(2, 1), MatrixCell::Wrong));
}

#[test]
fn difficulty_matrix_defaults_to_all_runs() {
    let s = store();
    let m = s.difficulty_matrix(&[]).unwrap();
    // Both fixture runs picked up.
    assert_eq!(m.runs.len(), 2);
}

#[test]
fn query_history_collects_per_run_answers() {
    let s = store();
    let h = s
        .query_history("1001", &["exp-a".into(), "exp-b".into()])
        .unwrap();
    assert_eq!(h.gold_answer, "Tagalog");
    assert!(h.question.contains("Philippines"));
    assert_eq!(h.entries.len(), 2);
    assert!(h.entries.iter().all(|e| e.exact));
    assert!(h.entries.iter().all(|e| e.agent_answer == "Tagalog"));
}

#[test]
fn query_history_marks_missing_runs() {
    // Stage a run dir that doesn't include query 1001.
    let tmp = tempfile::tempdir().unwrap();
    let runs = tmp.path().join("runs");
    std::fs::create_dir_all(runs.join("expM")).unwrap();
    let gold = tmp.path().join("gold.jsonl");
    std::fs::write(
        &gold,
        r#"{"query_id": "1001", "query": "q", "answer": "a"}"#,
    )
    .unwrap();
    let s = BenchStore::new(BenchPaths::new(runs, tmp.path().join("configs"), gold));
    let h = s.query_history("1001", &["expM".into()]).unwrap();
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0].status, "missing");
    assert!(!h.entries[0].exact);
}

#[test]
fn cost_summary_rolls_up_tokens() {
    let s = store();
    let rows = s.cost_summary(None).unwrap();
    assert_eq!(rows.len(), 2);
    let exp_a = rows.iter().find(|r| r.slug == "exp-a").unwrap();
    // exp-a: 5000 + 7000 + 4000 input = 16000; 500 + 800 + 400 output = 1700.
    assert_eq!(exp_a.input_tokens, 16_000);
    assert_eq!(exp_a.output_tokens, 1_700);
    // qwen3.5:9b is local — no cost estimate.
    assert!(exp_a.est_usd.is_none());
}

#[test]
fn cost_summary_window_filters() {
    // Window of 0 days excludes everything (mtimes are in the past).
    let s = store();
    let rows = s.cost_summary(Some(0)).unwrap();
    assert!(rows.is_empty());
}

#[test]
fn malformed_run_dir_is_skipped() {
    // The `_logs` subdirectory inside `runs/exp-a/` should not become
    // its own experiment, and the `_logs/.gitkeep` file inside the
    // experiment dir should not be treated as a run JSON.
    let s = store();
    let rows = s.list_experiments().unwrap();
    assert!(!rows.iter().any(|r| r.slug == "_logs"));
}
