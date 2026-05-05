//! `BenchStore` — read-mostly data layer behind the dashboard host
//! functions. Pure filesystem + JSON; no DB, no network.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use chrono::{DateTime, Utc};

use crate::paths::BenchPaths;
use crate::score::{extract_agent_answer, score_exact, score_fuzzy};
use crate::types::{
    Aggregate, Annotations, Comparison, ComparisonDeltas, ComparisonRow, ConfigInfo, CostRow,
    DifficultyMatrix, ExperimentRow, GoldEntry, MatrixCell, Note, QueryDetail, QueryHistory,
    QueryHistoryEntry, QuerySummary, RawGoldRow, RunDetail, RunRecord, Winner,
};

#[derive(thiserror::Error, Debug)]
pub enum BenchError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid path: {0}")]
    InvalidPath(String),
}

pub struct BenchStore {
    paths: BenchPaths,
    /// Lazily-loaded gold map, keyed by `query_id` as a string.
    gold_cache: OnceLock<HashMap<String, GoldEntry>>,
}

impl BenchStore {
    pub fn new(paths: BenchPaths) -> Self {
        Self {
            paths,
            gold_cache: OnceLock::new(),
        }
    }

    pub fn paths(&self) -> &BenchPaths {
        &self.paths
    }

    // -------------------------------------------------------------------- //
    // Gold answers (lazy-loaded once per store).
    // -------------------------------------------------------------------- //

    fn gold_map(&self) -> Result<&HashMap<String, GoldEntry>, BenchError> {
        if let Some(cached) = self.gold_cache.get() {
            return Ok(cached);
        }
        let map = load_gold_jsonl(&self.paths.gold_jsonl)?;
        // OnceLock::set returns Err if another thread won; either way we
        // can read via get() afterwards.
        let _ = self.gold_cache.set(map);
        Ok(self.gold_cache.get().expect("just set"))
    }

    pub fn gold(&self, query_id: &str) -> Result<Option<GoldEntry>, BenchError> {
        Ok(self.gold_map()?.get(query_id).cloned())
    }

    // -------------------------------------------------------------------- //
    // Annotations — sidecar `runs/<slug>/notes.json`.
    // -------------------------------------------------------------------- //

    pub fn read_annotations(&self, slug: &str) -> Result<Annotations, BenchError> {
        let path = self.notes_path(slug)?;
        if !path.exists() {
            return Ok(Annotations::default());
        }
        let raw = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Append a note and union the supplied tags into the sidecar. New
    /// tags preserve insertion order; duplicates against existing tags
    /// are dropped. Writes go through a tmp+rename so a crash mid-write
    /// can't leave a half-written `notes.json`.
    pub fn write_annotation(
        &self,
        slug: &str,
        text: &str,
        tags: Vec<String>,
    ) -> Result<Annotations, BenchError> {
        let path = self.notes_path(slug)?;
        let mut ann = self.read_annotations(slug)?;
        if !text.is_empty() {
            ann.notes.push(Note {
                ts: Utc::now(),
                text: text.to_string(),
            });
        }
        for tag in tags {
            if !ann.tags.contains(&tag) {
                ann.tags.push(tag);
            }
        }
        let tmp = path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&ann)?;
        fs::write(&tmp, body)?;
        fs::rename(&tmp, &path)?;
        Ok(ann)
    }

    fn notes_path(&self, slug: &str) -> Result<PathBuf, BenchError> {
        Ok(self.run_dir(slug)?.join("notes.json"))
    }

    // -------------------------------------------------------------------- //
    // list_experiments — one row per `runs/<slug>/`.
    // -------------------------------------------------------------------- //

    pub fn list_experiments(&self) -> Result<Vec<ExperimentRow>, BenchError> {
        let mut rows = Vec::new();
        if !self.paths.runs_dir.exists() {
            return Ok(rows);
        }
        for entry in fs::read_dir(&self.paths.runs_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let slug = match entry.file_name().into_string() {
                Ok(s) => s,
                Err(_) => continue,
            };
            // Skip dot-dirs and `_logs` siblings if any.
            if slug.starts_with('.') {
                continue;
            }
            match self.summarize_run(&slug) {
                Ok(row) => rows.push(row),
                Err(err) => {
                    tracing::warn!(slug = %slug, error = %err, "skipping run");
                }
            }
        }
        rows.sort_by(|a, b| b.mtime.cmp(&a.mtime));
        Ok(rows)
    }

    fn summarize_run(&self, slug: &str) -> Result<ExperimentRow, BenchError> {
        let detail = self.load_run(slug)?;
        let mtime = run_dir_mtime(&self.paths.runs_dir.join(slug))?;
        Ok(ExperimentRow {
            slug: slug.to_string(),
            n_queries: detail.aggregate.n_queries,
            exact: detail.aggregate.exact,
            fuzzy: detail.aggregate.fuzzy,
            p50_ms: detail.aggregate.p50_ms,
            mtime,
            tags: detail.tags,
        })
    }

    // -------------------------------------------------------------------- //
    // load_run — every per-query JSON in `runs/<slug>/`.
    // -------------------------------------------------------------------- //

    pub fn load_run(&self, slug: &str) -> Result<RunDetail, BenchError> {
        let dir = self.run_dir(slug)?;
        let gold = self.gold_map()?;

        let mut metadata = None;
        let mut queries: Vec<QuerySummary> = Vec::new();
        let mut total_input: u64 = 0;
        let mut total_output: u64 = 0;
        let mut total_ms: Vec<u64> = Vec::new();
        let mut exact_n = 0usize;
        let mut fuzzy_n = 0usize;

        for path in run_files(&dir)? {
            let raw = fs::read_to_string(&path)?;
            let record: RunRecord = match serde_json::from_str(&raw) {
                Ok(r) => r,
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "skipping malformed run JSON");
                    continue;
                }
            };
            // First good record stamps the run's metadata.
            if metadata.is_none() {
                metadata = Some(record.metadata.clone());
            }
            let qid = record.query_id.clone().unwrap_or_default();
            let agent_answer = extract_agent_answer(&record.result);
            let gold_answer = gold.get(&qid).map(|g| g.answer.as_str()).unwrap_or("");
            let exact = score_exact(&agent_answer, gold_answer);
            let fuzzy = score_fuzzy(&agent_answer, gold_answer);
            if exact {
                exact_n += 1;
            }
            if fuzzy {
                fuzzy_n += 1;
            }
            total_input += record.usage.input_tokens as u64;
            total_output += record.usage.output_tokens as u64;
            let q_ms = record.timing.as_ref().map(|t| t.total_ms);
            if let Some(ms) = q_ms {
                total_ms.push(ms);
            }
            queries.push(QuerySummary {
                query_id: qid,
                status: record.status,
                exact,
                fuzzy,
                iterations: record.iterations_used,
                total_ms: q_ms,
                total_tokens: record.usage.total_tokens,
                termination_reason: record.termination_reason,
            });
        }

        queries.sort_by(|a, b| a.query_id.cmp(&b.query_id));

        let aggregate = Aggregate {
            n_queries: queries.len(),
            exact: exact_n,
            fuzzy: fuzzy_n,
            p50_ms: percentile(&mut total_ms.clone(), 50),
            p95_ms: percentile(&mut total_ms, 95),
            total_input_tokens: total_input,
            total_output_tokens: total_output,
        };

        let ann = self.read_annotations(slug).unwrap_or_default();

        Ok(RunDetail {
            slug: slug.to_string(),
            metadata,
            queries,
            aggregate,
            notes: ann.notes,
            tags: ann.tags,
        })
    }

    // -------------------------------------------------------------------- //
    // load_query — full record for one (slug, query_id).
    // -------------------------------------------------------------------- //

    pub fn load_query(&self, slug: &str, query_id: &str) -> Result<QueryDetail, BenchError> {
        let dir = self.run_dir(slug)?;
        let gold = self.gold_map()?;
        for path in run_files(&dir)? {
            let raw = fs::read_to_string(&path)?;
            let record: RunRecord = match serde_json::from_str(&raw) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if record.query_id.as_deref() != Some(query_id) {
                continue;
            }
            let agent_answer = extract_agent_answer(&record.result);
            let (gold_answer, question) = match gold.get(query_id) {
                Some(g) => (g.answer.clone(), g.query.clone()),
                None => (String::new(), String::new()),
            };
            let exact = score_exact(&agent_answer, &gold_answer);
            let fuzzy = score_fuzzy(&agent_answer, &gold_answer);
            return Ok(QueryDetail {
                slug: slug.to_string(),
                query_id: query_id.to_string(),
                question,
                gold_answer,
                agent_answer,
                exact,
                fuzzy,
                trace: record.result,
                trajectory: record.trajectory,
                retrieved_docids: record.retrieved_docids,
                timing: record.timing,
                usage: record.usage,
                iterations: record.iterations_used,
                termination_reason: record.termination_reason,
            });
        }
        Err(BenchError::NotFound(format!("{slug}/{query_id}")))
    }

    // -------------------------------------------------------------------- //
    // load_config — TOML at `bench/browsecomp/configs/<slug>.toml`.
    // -------------------------------------------------------------------- //

    pub fn load_config(&self, slug: &str) -> Result<ConfigInfo, BenchError> {
        // Slug → config name has been informal historically (the slug
        // sometimes prefixes a date). Try the literal slug first, then
        // strip leading date-prefix tokens like `apr19-`.
        let mut candidates: Vec<String> = vec![format!("{slug}.toml")];
        if let Some(rest) = slug.split_once('-').map(|(_, r)| r) {
            candidates.push(format!("{rest}.toml"));
        }
        for name in &candidates {
            let path = self.paths.configs_dir.join(name);
            if path.exists() {
                let contents = fs::read_to_string(&path)?;
                return Ok(ConfigInfo {
                    slug: slug.to_string(),
                    toml_path: Some(path.to_string_lossy().into_owned()),
                    contents: Some(contents),
                });
            }
        }
        Ok(ConfigInfo {
            slug: slug.to_string(),
            toml_path: None,
            contents: None,
        })
    }

    // -------------------------------------------------------------------- //
    // compare — query-by-query classification of two runs.
    // -------------------------------------------------------------------- //

    pub fn compare(&self, slug_a: &str, slug_b: &str) -> Result<Comparison, BenchError> {
        let a = self.load_run(slug_a)?;
        let b = self.load_run(slug_b)?;
        let map_a: HashMap<String, QuerySummary> = a
            .queries
            .iter()
            .map(|q| (q.query_id.clone(), q.clone()))
            .collect();
        let map_b: HashMap<String, QuerySummary> = b
            .queries
            .iter()
            .map(|q| (q.query_id.clone(), q.clone()))
            .collect();

        let mut all_qids: Vec<String> = map_a.keys().chain(map_b.keys()).cloned().collect();
        all_qids.sort();
        all_qids.dedup();

        let mut per_query = Vec::with_capacity(all_qids.len());
        for qid in all_qids {
            let qa = map_a.get(&qid).cloned();
            let qb = map_b.get(&qid).cloned();
            let winner = classify_winner(qa.as_ref(), qb.as_ref());
            per_query.push(ComparisonRow {
                query_id: qid,
                a: qa,
                b: qb,
                winner,
            });
        }

        let deltas = ComparisonDeltas {
            exact_a: a.aggregate.exact,
            exact_b: b.aggregate.exact,
            fuzzy_a: a.aggregate.fuzzy,
            fuzzy_b: b.aggregate.fuzzy,
            p50_ms_a: a.aggregate.p50_ms,
            p50_ms_b: b.aggregate.p50_ms,
            total_tokens_a: a.aggregate.total_input_tokens + a.aggregate.total_output_tokens,
            total_tokens_b: b.aggregate.total_input_tokens + b.aggregate.total_output_tokens,
        };

        Ok(Comparison {
            slug_a: slug_a.to_string(),
            slug_b: slug_b.to_string(),
            per_query,
            deltas,
        })
    }

    // -------------------------------------------------------------------- //
    // difficulty_matrix — N runs × union(query_ids) grid of cells.
    // -------------------------------------------------------------------- //

    /// If `slugs` is empty, defaults to every experiment in `runs/`.
    pub fn difficulty_matrix(&self, slugs: &[String]) -> Result<DifficultyMatrix, BenchError> {
        let resolved: Vec<String> = if slugs.is_empty() {
            self.list_experiments()?
                .into_iter()
                .map(|r| r.slug)
                .collect()
        } else {
            slugs.to_vec()
        };

        // Load each run once, indexing its queries by id.
        let mut per_run: Vec<HashMap<String, QuerySummary>> = Vec::with_capacity(resolved.len());
        for slug in &resolved {
            let detail = self.load_run(slug)?;
            let map = detail
                .queries
                .into_iter()
                .map(|q| (q.query_id.clone(), q))
                .collect();
            per_run.push(map);
        }

        // Union of query ids across all runs, sorted lexicographically.
        let mut all_qids: Vec<String> = per_run
            .iter()
            .flat_map(|m| m.keys().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        all_qids.sort();

        let mut cells: Vec<Vec<MatrixCell>> = Vec::with_capacity(all_qids.len());
        for qid in &all_qids {
            let mut row = Vec::with_capacity(per_run.len());
            for run in &per_run {
                row.push(classify_cell(run.get(qid)));
            }
            cells.push(row);
        }

        Ok(DifficultyMatrix {
            queries: all_qids,
            runs: resolved,
            cells,
        })
    }

    // -------------------------------------------------------------------- //
    // query_history — one query, every run.
    // -------------------------------------------------------------------- //

    pub fn query_history(
        &self,
        query_id: &str,
        slugs: &[String],
    ) -> Result<QueryHistory, BenchError> {
        let resolved: Vec<String> = if slugs.is_empty() {
            self.list_experiments()?
                .into_iter()
                .map(|r| r.slug)
                .collect()
        } else {
            slugs.to_vec()
        };

        let gold = self.gold_map()?;
        let (question, gold_answer) = match gold.get(query_id) {
            Some(g) => (g.query.clone(), g.answer.clone()),
            None => (String::new(), String::new()),
        };

        let mut entries = Vec::with_capacity(resolved.len());
        for slug in resolved {
            let entry = match self.load_query(&slug, query_id) {
                Ok(detail) => QueryHistoryEntry {
                    slug,
                    status: "completed".into(),
                    exact: detail.exact,
                    fuzzy: detail.fuzzy,
                    agent_answer: detail.agent_answer,
                    retrieved_docids: detail.retrieved_docids,
                    iterations: detail.iterations,
                    total_ms: detail.timing.as_ref().map(|t| t.total_ms),
                },
                Err(BenchError::NotFound(_)) => QueryHistoryEntry {
                    slug,
                    status: "missing".into(),
                    exact: false,
                    fuzzy: false,
                    agent_answer: String::new(),
                    retrieved_docids: Vec::new(),
                    iterations: 0,
                    total_ms: None,
                },
                Err(err) => return Err(err),
            };
            entries.push(entry);
        }

        Ok(QueryHistory {
            query_id: query_id.to_string(),
            question,
            gold_answer,
            entries,
        })
    }

    // -------------------------------------------------------------------- //
    // cost_summary — token rollup per run, optionally windowed.
    // -------------------------------------------------------------------- //

    /// `window_days = Some(n)` filters to runs touched in the last `n`
    /// days; `None` returns all runs. Sorted newest-first.
    pub fn cost_summary(&self, window_days: Option<u32>) -> Result<Vec<CostRow>, BenchError> {
        let cutoff: Option<DateTime<Utc>> =
            window_days.map(|n| Utc::now() - chrono::Duration::days(n as i64));

        let mut out = Vec::new();
        for row in self.list_experiments()? {
            if let Some(cutoff_ts) = cutoff {
                if row.mtime < cutoff_ts {
                    continue;
                }
            }
            let detail = self.load_run(&row.slug)?;
            let model = detail
                .metadata
                .as_ref()
                .map(|m| m.model.clone())
                .unwrap_or_default();
            let input = detail.aggregate.total_input_tokens;
            let output = detail.aggregate.total_output_tokens;
            out.push(CostRow {
                slug: row.slug,
                model: model.clone(),
                input_tokens: input,
                output_tokens: output,
                est_usd: estimate_usd(&model, input, output),
                mtime: row.mtime,
            });
        }
        out.sort_by(|a, b| b.mtime.cmp(&a.mtime));
        Ok(out)
    }

    // -------------------------------------------------------------------- //
    // Helpers.
    // -------------------------------------------------------------------- //

    fn run_dir(&self, slug: &str) -> Result<PathBuf, BenchError> {
        if slug.is_empty() || slug.contains('/') || slug.contains("..") {
            return Err(BenchError::InvalidPath(slug.to_string()));
        }
        let dir = self.paths.runs_dir.join(slug);
        if !dir.exists() {
            return Err(BenchError::NotFound(format!("runs/{slug}")));
        }
        Ok(dir)
    }
}

/// Cell classification is correctness-first: timeouts and
/// max-turns-fallback runs that still extracted a correct answer
/// count as Exact / Fuzzy. Only an explicit `status == "error"`
/// hides correctness. The UI can layer a "was-timeout" indicator
/// using `QuerySummary::termination_reason` separately.
fn classify_cell(q: Option<&QuerySummary>) -> MatrixCell {
    let Some(q) = q else {
        return MatrixCell::Missing;
    };
    if q.status == "error" {
        return MatrixCell::Error;
    }
    if q.exact {
        MatrixCell::Exact
    } else if q.fuzzy {
        MatrixCell::Fuzzy
    } else {
        MatrixCell::Wrong
    }
}

/// Placeholder pricing — local Ollama is free, OpenAI rough estimates.
/// Replace with a config-driven table when we have stable rates to
/// commit. Returns `None` for unknown models so the UI can render `—`.
fn estimate_usd(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    let (in_per_m, out_per_m): (f64, f64) = if model.starts_with("gpt-5") {
        (10.0, 40.0)
    } else if model.starts_with("gpt-4") {
        (2.5, 10.0)
    } else {
        return None;
    };
    Some(
        (input_tokens as f64) * in_per_m / 1_000_000.0
            + (output_tokens as f64) * out_per_m / 1_000_000.0,
    )
}

fn classify_winner(a: Option<&QuerySummary>, b: Option<&QuerySummary>) -> Winner {
    match (a, b) {
        (None, None) => Winner::Tie,
        (Some(_), None) => Winner::OnlyA,
        (None, Some(_)) => Winner::OnlyB,
        (Some(qa), Some(qb)) => match (qa.exact, qb.exact) {
            (true, true) => Winner::BothCorrect,
            (true, false) => Winner::A,
            (false, true) => Winner::B,
            (false, false) => match (qa.fuzzy, qb.fuzzy) {
                (true, false) => Winner::A,
                (false, true) => Winner::B,
                (true, true) => Winner::BothCorrect,
                (false, false) => Winner::BothWrong,
            },
        },
    }
}

fn percentile(values: &mut [u64], p: u8) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let idx = ((values.len() as f64) * (p as f64) / 100.0).ceil() as usize;
    let idx = idx.saturating_sub(1).min(values.len() - 1);
    Some(values[idx])
}

fn run_files(dir: &Path) -> Result<Vec<PathBuf>, BenchError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // Convention: per-query files start with `run_` and end with `.json`.
        if name.starts_with("run_") && name.ends_with(".json") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn run_dir_mtime(dir: &Path) -> Result<DateTime<Utc>, BenchError> {
    let metadata = fs::metadata(dir)?;
    let modified = metadata.modified()?;
    Ok(DateTime::<Utc>::from(modified))
}

fn load_gold_jsonl(path: &Path) -> Result<HashMap<String, GoldEntry>, BenchError> {
    let mut map = HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    let raw = fs::read_to_string(path)?;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: RawGoldRow = serde_json::from_str(line)?;
        let qid = match &row.query_id {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => continue,
        };
        map.insert(
            qid.clone(),
            GoldEntry {
                query_id: qid,
                query: row.query,
                answer: row.answer,
            },
        );
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_basic() {
        let v = vec![100, 200, 300, 400, 500];
        assert_eq!(percentile(&mut v.clone(), 50), Some(300));
        assert_eq!(percentile(&mut v.clone(), 95), Some(500));
        assert_eq!(percentile(&mut Vec::<u64>::new(), 50), None);
    }

    #[test]
    fn classify_winner_branches() {
        let q = |exact: bool, fuzzy: bool| QuerySummary {
            query_id: "1".into(),
            status: "completed".into(),
            exact,
            fuzzy,
            iterations: 1,
            total_ms: None,
            total_tokens: 0,
            termination_reason: "final_called".into(),
        };
        let a = q(true, true);
        let b = q(false, false);
        assert!(matches!(classify_winner(Some(&a), Some(&b)), Winner::A));
        assert!(matches!(
            classify_winner(Some(&q(false, true)), Some(&q(false, false))),
            Winner::A
        ));
        assert!(matches!(
            classify_winner(Some(&q(false, false)), Some(&q(false, false))),
            Winner::BothWrong
        ));
        assert!(matches!(classify_winner(None, Some(&b)), Winner::OnlyB));
    }
}
