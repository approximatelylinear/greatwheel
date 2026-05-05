//! Smoke test: list experiments and load one run against the real
//! workspace's `runs/` directory. Verifies the BenchStore handles
//! actual production data without crashing.
//!
//! Run from workspace root:
//!     cargo run -p gw-bench-ui --example smoke

use std::path::PathBuf;

use gw_bench_ui::{BenchPaths, BenchStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let paths = BenchPaths::from_workspace_root(&root);
    println!("workspace root: {}", root.display());
    println!("runs: {}", paths.runs_dir.display());
    println!("gold: {}", paths.gold_jsonl.display());

    let store = BenchStore::new(paths);
    let rows = store.list_experiments()?;
    println!("\nfound {} experiments:", rows.len());
    for row in rows.iter().take(10) {
        println!(
            "  {:36} n={:>3}  exact={:>2}  fuzzy={:>2}  p50={}",
            row.slug,
            row.n_queries,
            row.exact,
            row.fuzzy,
            row.p50_ms
                .map(|ms| format!("{ms}ms"))
                .unwrap_or_else(|| "—".into()),
        );
    }

    if let Some(first) = rows.first() {
        let detail = store.load_run(&first.slug)?;
        println!(
            "\nfirst run detail: slug={} model={:?}",
            detail.slug,
            detail.metadata.as_ref().map(|m| &m.model)
        );
        println!(
            "  aggregate: exact={}/{} fuzzy={}/{} p50={:?}",
            detail.aggregate.exact,
            detail.aggregate.n_queries,
            detail.aggregate.fuzzy,
            detail.aggregate.n_queries,
            detail.aggregate.p50_ms,
        );
    }

    // Difficulty matrix over the top-3 most recent runs.
    let top_slugs: Vec<String> = rows.iter().take(3).map(|r| r.slug.clone()).collect();
    if !top_slugs.is_empty() {
        let m = store.difficulty_matrix(&top_slugs)?;
        println!(
            "\ndifficulty matrix: {} queries × {} runs",
            m.queries.len(),
            m.runs.len()
        );
        let mut tally: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for row in &m.cells {
            for cell in row {
                let key = match cell {
                    gw_bench_ui::types::MatrixCell::Exact => "exact",
                    gw_bench_ui::types::MatrixCell::Fuzzy => "fuzzy",
                    gw_bench_ui::types::MatrixCell::Wrong => "wrong",
                    gw_bench_ui::types::MatrixCell::Error => "error",
                    gw_bench_ui::types::MatrixCell::Missing => "missing",
                };
                *tally.entry(key).or_default() += 1;
            }
        }
        for (k, v) in tally {
            println!("  {k:>8}: {v}");
        }
    }

    // Cost summary — top 5 most recent.
    let costs = store.cost_summary(None)?;
    println!("\ncost summary (top 5):");
    for row in costs.iter().take(5) {
        println!(
            "  {:36}  in={:>9}  out={:>7}  ${}",
            row.slug,
            row.input_tokens,
            row.output_tokens,
            row.est_usd
                .map(|u| format!("{u:.4}"))
                .unwrap_or_else(|| "—".into()),
        );
    }
    Ok(())
}
