//! Phase 3 — per-topic LLM synthesis.
//!
//! For each rich topic (chunk_count ≥ min_chunks), generate a 2–4 paragraph
//! summary that integrates information across its member chunks. The
//! summary is persisted to `kb_topics.summary` and is what the user
//! actually reads when browsing a topic.
//!
//! ### Source-balanced chunk selection
//!
//! The naive "top N chunks by relevance" is wrong for topics whose
//! membership is skewed across sources. A topic with 80 chunks from one
//! source and 10 from another would get a prompt dominated by the noisy
//! source, defeating the cross-source integration that makes synthesis
//! valuable. `select_balanced_chunks` takes the top `budget / n_sources`
//! per source first, then fills the remainder with the highest-relevance
//! chunks that weren't already picked.
//!
//! ### Mechanical source attribution
//!
//! We deliberately do not ask the LLM to cite sources. The model would
//! produce inconsistent citation formats ([1], [Source A], "according
//! to..."), and inline citations are hard to parse. Instead we append a
//! "Sources:" footer that lists the distinct source titles we actually
//! fed into the prompt. The attribution is deterministic and trustworthy.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use gw_llm::{Message, OllamaClient};
use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;

/// Character budget per chunk in the prompt.
const CHUNK_CHAR_LIMIT: usize = 500;
/// Maximum total chunks fed to the synthesizer per topic.
const MAX_CHUNKS_PER_PROMPT: usize = 20;

const SYSTEM_PROMPT: &str = "You write concise, integrative summaries of \
knowledge-base topics. Output plain prose only — no headers, no bullet \
lists, no markdown, no preamble, no quoted passages.";

#[derive(Debug, Clone, Copy)]
pub struct SynthesizeOpts {
    /// Process at most this many topics.
    pub limit: Option<usize>,
    /// Only topics with chunk_count ≥ this will be summarized.
    pub min_chunks: i32,
    /// If true, regenerate summaries that already exist.
    pub regenerate: bool,
    /// If true, regenerate only summaries whose `summary_at < updated_at`.
    pub stale_only: bool,
    /// If set, restrict the run to a single topic regardless of other
    /// filters. Bypasses `min_chunks`, `regenerate`, and `stale_only`.
    pub only_topic: Option<Uuid>,
}

impl Default for SynthesizeOpts {
    fn default() -> Self {
        Self {
            limit: None,
            min_chunks: 5,
            regenerate: false,
            stale_only: false,
            only_topic: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SynthesizeReport {
    pub topics_considered: usize,
    pub topics_skipped: usize,
    pub summaries_written: usize,
    pub llm_failures: usize,
}

/// Walk rich topics and synthesize summaries for the ones that need one.
pub async fn synthesize_topics(
    stores: &KbStores,
    opts: SynthesizeOpts,
) -> Result<SynthesizeReport, KbError> {
    let mut report = SynthesizeReport::default();

    let candidates = fetch_candidates(&stores.pg, &opts).await?;
    info!(
        candidates = candidates.len(),
        min_chunks = opts.min_chunks,
        regenerate = opts.regenerate,
        stale_only = opts.stale_only,
        "loaded candidate topics"
    );
    report.topics_considered = candidates.len();

    let limit = opts.limit.unwrap_or(usize::MAX);
    for (idx, topic) in candidates.iter().take(limit).enumerate() {
        info!(
            n = idx + 1,
            total = candidates.len().min(limit),
            label = %topic.label,
            chunks = topic.chunk_count,
            "synthesizing"
        );

        let chunks = select_balanced_chunks(&stores.pg, topic.topic_id).await?;
        if chunks.is_empty() {
            warn!(label = %topic.label, "no chunks for topic, skipping");
            report.topics_skipped += 1;
            continue;
        }

        let summary = match call_synthesizer(&stores.llm, topic, &chunks).await {
            Ok(s) => s,
            Err(e) => {
                warn!(err = %e, "synthesizer failed, skipping topic");
                report.llm_failures += 1;
                continue;
            }
        };

        // Mechanical source attribution — list the distinct sources we
        // actually fed in, alphabetized for stability.
        let mut source_titles: Vec<String> = chunks
            .iter()
            .map(|c| c.source_title.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        source_titles.sort();
        let sources_footer = format!("\n\nSources: {}", source_titles.join("; "));
        let final_summary = format!("{}{}", summary.trim(), sources_footer);

        persist_summary(&stores.pg, topic.topic_id, &final_summary).await?;
        report.summaries_written += 1;
        debug!(label = %topic.label, "summary persisted");
    }

    info!(
        written = report.summaries_written,
        failures = report.llm_failures,
        "synthesize complete"
    );
    Ok(report)
}

#[derive(Debug, Clone)]
struct CandidateTopic {
    topic_id: Uuid,
    label: String,
    chunk_count: i32,
}

async fn fetch_candidates(
    pool: &PgPool,
    opts: &SynthesizeOpts,
) -> Result<Vec<CandidateTopic>, KbError> {
    // Single-topic mode bypasses all other filters.
    if let Some(target) = opts.only_topic {
        let row: Option<(Uuid, String, i32)> = sqlx::query_as(
            "SELECT topic_id, label, chunk_count FROM kb_topics WHERE topic_id = $1",
        )
        .bind(target)
        .fetch_optional(pool)
        .await?;
        return Ok(row
            .into_iter()
            .map(|(id, label, cc)| CandidateTopic {
                topic_id: id,
                label,
                chunk_count: cc,
            })
            .collect());
    }

    // Build filter combinations. `regenerate` wins over everything else.
    // Otherwise: `stale_only` OR `summary IS NULL` gates whether a topic
    // gets picked up.
    let regenerate = opts.regenerate;
    let stale_only = opts.stale_only;
    let min_chunks = opts.min_chunks;

    let rows: Vec<(Uuid, String, i32)> = sqlx::query_as(
        r#"
        SELECT topic_id, label, chunk_count
        FROM kb_topics
        WHERE chunk_count >= $1
          AND (
                $2                     -- regenerate: take everything
             OR ($3 AND (summary_at IS NULL OR summary_at < updated_at))
             OR (NOT $2 AND NOT $3 AND summary IS NULL)
          )
        ORDER BY chunk_count DESC, label ASC
        "#,
    )
    .bind(min_chunks)
    .bind(regenerate)
    .bind(stale_only)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, label, cc)| CandidateTopic {
            topic_id: id,
            label,
            chunk_count: cc,
        })
        .collect())
}

#[derive(Debug, Clone)]
struct SelectedChunk {
    source_title: String,
    heading_path: Vec<String>,
    content: String,
}

/// Source-balanced chunk selection. See module docs.
async fn select_balanced_chunks(
    pool: &PgPool,
    topic_id: Uuid,
) -> Result<Vec<SelectedChunk>, KbError> {
    // Fetch all member chunks with their source info, ordered by (source,
    // relevance desc) so we can walk them in a single pass.
    #[derive(sqlx::FromRow)]
    struct Row {
        source_id: Uuid,
        source_title: String,
        heading_path: Vec<String>,
        content: String,
        relevance: f32,
    }

    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT c.source_id,
               s.title         AS source_title,
               c.heading_path,
               c.content,
               tc.relevance
        FROM kb_topic_chunks tc
        JOIN kb_chunks c USING (chunk_id)
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE tc.topic_id = $1
        ORDER BY c.source_id, tc.relevance DESC
        "#,
    )
    .bind(topic_id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Pass 1: group by source, walking the sorted list.
    let mut by_source: Vec<(Uuid, Vec<Row>)> = Vec::new();
    for row in rows {
        if let Some((sid, group)) = by_source.last_mut() {
            if *sid == row.source_id {
                group.push(row);
                continue;
            }
        }
        by_source.push((row.source_id, vec![row]));
    }

    let n_sources = by_source.len();
    let budget = MAX_CHUNKS_PER_PROMPT;
    let per_source = (budget / n_sources).max(1);

    // Pass 2: take the top `per_source` from each source.
    let mut selected: Vec<Row> = Vec::with_capacity(budget);
    let mut leftover: Vec<Row> = Vec::new();
    for (_sid, mut group) in by_source {
        let take = per_source.min(group.len());
        let head: Vec<Row> = group.drain(..take).collect();
        selected.extend(head);
        leftover.extend(group);
    }

    // Pass 3: fill remaining budget with highest-relevance leftovers.
    if selected.len() < budget {
        leftover.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let need = budget - selected.len();
        selected.extend(leftover.into_iter().take(need));
    }
    selected.truncate(budget);

    // Final ordering: by source, then relevance, so related passages
    // appear together in the prompt.
    selected.sort_by(|a, b| {
        a.source_title.cmp(&b.source_title).then(
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    Ok(selected
        .into_iter()
        .map(|r| SelectedChunk {
            source_title: r.source_title,
            heading_path: r.heading_path,
            content: r.content,
        })
        .collect())
}

async fn call_synthesizer(
    llm: &OllamaClient,
    topic: &CandidateTopic,
    chunks: &[SelectedChunk],
) -> Result<String, KbError> {
    let prompt = build_prompt(topic, chunks);
    let messages = vec![
        Message {
            role: "system".into(),
            content: SYSTEM_PROMPT.into(),
        },
        Message {
            role: "user".into(),
            content: prompt,
        },
    ];
    let resp = llm
        .chat_with_options(&messages, None, Some(false))
        .await
        .map_err(|e| KbError::Other(format!("llm chat: {e}")))?;
    let cleaned = resp.content.trim().to_string();
    if cleaned.is_empty() {
        return Err(KbError::Other("synthesizer returned empty content".into()));
    }
    Ok(cleaned)
}

fn build_prompt(topic: &CandidateTopic, chunks: &[SelectedChunk]) -> String {
    let n_sources: usize = chunks
        .iter()
        .map(|c| c.source_title.as_str())
        .collect::<HashSet<_>>()
        .len();

    let mut body = String::new();
    for (i, c) in chunks.iter().enumerate() {
        let heading = if c.heading_path.is_empty() {
            String::from("(root)")
        } else {
            c.heading_path.join(" > ")
        };
        let excerpt: String = c
            .content
            .chars()
            .take(CHUNK_CHAR_LIMIT)
            .collect::<String>()
            .replace('\n', " ");
        body.push_str(&format!(
            "\n[passage {}] source={:?} section={:?}\n{}\n",
            i + 1,
            c.source_title,
            heading,
            excerpt.trim()
        ));
    }

    format!(
        "Write a 2-4 paragraph synthesis of the knowledge-base topic below.

Topic: {label}

The {n_chunks} passages below come from {n_sources} different sources. \
Integrate the information across them into coherent prose: distill the \
key facts, findings, and claims; note where sources agree, disagree, \
or offer different angles; capture evolution or chronology where \
relevant. Do NOT simply enumerate or paraphrase each passage one by \
one — the point is integration. Use your own words; do not quote \
directly. Do not cite sources inline — just produce the synthesis as \
plain prose. No headers, no bullet points, no markdown fences.

{body}

Output the synthesis now as 2-4 paragraphs of plain prose:",
        label = topic.label,
        n_chunks = chunks.len(),
    )
}

async fn persist_summary(pool: &PgPool, topic_id: Uuid, summary: &str) -> Result<(), KbError> {
    sqlx::query(
        r#"
        UPDATE kb_topics
        SET summary = $1,
            summary_at = now()
        WHERE topic_id = $2
        "#,
    )
    .bind(summary)
    .bind(topic_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch the currently-stored summary for a topic, if any. Used by
/// `gw-kb topic <slug>` for display.
pub async fn fetch_summary(
    pool: &PgPool,
    topic_id: Uuid,
) -> Result<Option<(String, DateTime<Utc>)>, KbError> {
    let row: Option<(Option<String>, Option<DateTime<Utc>>)> =
        sqlx::query_as("SELECT summary, summary_at FROM kb_topics WHERE topic_id = $1")
            .bind(topic_id)
            .fetch_optional(pool)
            .await?;
    Ok(row.and_then(|(s, at)| match (s, at) {
        (Some(s), Some(at)) => Some((s, at)),
        _ => None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(source: &str) -> SelectedChunk {
        SelectedChunk {
            source_title: source.to_string(),
            heading_path: vec![],
            content: "content".into(),
        }
    }

    #[test]
    fn prompt_counts_distinct_sources() {
        let topic = CandidateTopic {
            topic_id: Uuid::nil(),
            label: "Test".into(),
            chunk_count: 3,
        };
        let chunks = vec![row("Source A"), row("Source A"), row("Source B")];
        let p = build_prompt(&topic, &chunks);
        assert!(p.contains("3 passages"));
        assert!(p.contains("2 different sources"));
    }
}
