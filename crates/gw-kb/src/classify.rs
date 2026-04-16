//! LLM-based edge classifier — the typed-edges layer on top of `linking.rs`.
//!
//! Walks every edge in `kb_topic_links` (or only `related` ones, by default
//! for idempotency) and asks the tagger LLM to choose between
//! `subtopic_of`, `builds_on`, `contradicts`, or `related`. The prompt
//! includes both topic labels plus a representative chunk excerpt from
//! each topic, so the model has grounding beyond the label strings.
//!
//! ### Direction handling (the comment we promised in design)
//!
//! `kb_topic_links` stores each undirected pair exactly once, with the
//! lower UUID as `from_topic_id` (the convention used by `linking.rs`).
//! For *directional* kinds (`subtopic_of`, `builds_on`), we OVERRIDE this
//! convention so the row's `(from_topic_id, to_topic_id)` columns reflect
//! the semantic direction:
//!
//!   - `subtopic_of`: `from` is the more specific topic, `to` is the broader
//!     parent (i.e. the row reads "from is_a to")
//!   - `builds_on`:   `from` is the dependent / extension topic, `to` is the
//!     prerequisite that came first (i.e. the row reads "from builds_on to")
//!
//! For symmetric kinds (`related`, `contradicts`) the lower-UUID-first
//! convention is preserved. The bidirectional CTE walker in
//! `linking::spread_from_seeds` is unaffected because it unions both
//! directions of every row regardless.

use std::collections::HashMap;

use gw_llm::{Message, OllamaClient};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;

const REP_CHUNK_CHARS: usize = 220;
const SYSTEM_PROMPT: &str = "You classify the relationship between two topics \
in a knowledge base. Output JSON only — no prose, no markdown fences.";

#[derive(Debug, Clone, Copy, Default)]
pub struct ClassifyOpts {
    /// Process at most this many edges.
    pub limit: Option<usize>,
    /// If false, skip edges whose `kind` is already not `related`.
    pub reclassify: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ClassifyReport {
    pub edges_seen: usize,
    pub edges_skipped: usize,
    pub edges_classified: usize,
    pub edges_directional: usize,
    pub edges_contradicts: usize,
    pub edges_related: usize,
    pub llm_failures: usize,
    pub by_kind: HashMap<String, usize>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct ClassifierOutput {
    /// "related" | "subtopic_of" | "builds_on" | "contradicts"
    #[serde(default)]
    kind: String,
    /// "A" | "B" — only meaningful for directional kinds. The model
    /// often emits `null` here for symmetric kinds, so we accept missing
    /// or null and default to empty string.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    from: String,
}

fn deserialize_optional_string<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(d)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Debug, Clone)]
struct TopicSnapshot {
    label: String,
    excerpt: String,
}

/// Classify every edge in `kb_topic_links` according to `opts`.
pub async fn classify_edges(
    stores: &KbStores,
    opts: ClassifyOpts,
) -> Result<ClassifyReport, KbError> {
    let mut report = ClassifyReport::default();

    // 1. Load a representative excerpt per topic in one query.
    let snapshots = load_topic_snapshots(&stores.pg).await?;
    info!(topics = snapshots.len(), "loaded topic snapshots");

    // 2. Pull the edges we're going to process.
    let edge_filter = if opts.reclassify {
        ""
    } else {
        "WHERE kind = 'related'"
    };
    let limit_sql = match opts.limit {
        Some(n) => format!("LIMIT {}", n),
        None => String::new(),
    };
    let query = format!(
        "SELECT from_topic_id, to_topic_id, kind::text, confidence
         FROM kb_topic_links
         {edge_filter}
         ORDER BY confidence DESC
         {limit_sql}"
    );
    let edges: Vec<(Uuid, Uuid, String, f32)> =
        sqlx::query_as(&query).fetch_all(&stores.pg).await?;
    info!(edges = edges.len(), "edges to classify");

    // 3. Loop, classify, persist.
    for (idx, (from_id, to_id, _old_kind, confidence)) in edges.iter().enumerate() {
        report.edges_seen += 1;
        let Some(snap_a) = snapshots.get(from_id) else {
            warn!(topic = %from_id, "no snapshot for topic, skipping");
            report.edges_skipped += 1;
            continue;
        };
        let Some(snap_b) = snapshots.get(to_id) else {
            warn!(topic = %to_id, "no snapshot for topic, skipping");
            report.edges_skipped += 1;
            continue;
        };

        info!(
            n = idx + 1,
            total = edges.len(),
            a = %snap_a.label,
            b = %snap_b.label,
            "classifying"
        );

        let parsed = match classify_pair(&stores.llm, snap_a, snap_b).await {
            Ok(p) => p,
            Err(e) => {
                warn!(err = %e, "classifier call failed, leaving edge as 'related'");
                report.llm_failures += 1;
                continue;
            }
        };

        let kind = parsed.kind.trim().to_lowercase();
        let kind = match kind.as_str() {
            "subtopic_of" | "builds_on" | "contradicts" | "related" => kind,
            other => {
                warn!(
                    kind = other,
                    "unknown kind from classifier, defaulting to related"
                );
                "related".to_string()
            }
        };

        debug!(kind = %kind, from = %parsed.from, "classifier output");

        // Direction handling: rewrite the row so (from, to) reflects semantic
        // direction for directional kinds. For symmetric kinds we leave the
        // existing lower-UUID-first ordering alone.
        let directional = matches!(kind.as_str(), "subtopic_of" | "builds_on");
        let (new_from, new_to) = if directional {
            match parsed.from.trim() {
                "B" | "b" => (*to_id, *from_id),
                _ => (*from_id, *to_id), // default to A as `from`
            }
        } else {
            (*from_id, *to_id)
        };

        if directional {
            report.edges_directional += 1;
        } else if kind == "contradicts" {
            report.edges_contradicts += 1;
        } else {
            report.edges_related += 1;
        }
        *report.by_kind.entry(kind.clone()).or_insert(0) += 1;
        report.edges_classified += 1;

        // Persist. If direction was swapped, we may need to delete-then-insert
        // because the (from, to) primary key changed. Postgres allows updating
        // a primary key in-place as long as no other row holds the new key.
        // Since each undirected pair has exactly one row, the swap is safe.
        if (new_from, new_to) != (*from_id, *to_id) {
            sqlx::query(
                "UPDATE kb_topic_links
                 SET from_topic_id = $1, to_topic_id = $2, kind = $3::kb_link_kind
                 WHERE from_topic_id = $4 AND to_topic_id = $5",
            )
            .bind(new_from)
            .bind(new_to)
            .bind(&kind)
            .bind(from_id)
            .bind(to_id)
            .execute(&stores.pg)
            .await?;
        } else {
            sqlx::query(
                "UPDATE kb_topic_links
                 SET kind = $1::kb_link_kind
                 WHERE from_topic_id = $2 AND to_topic_id = $3",
            )
            .bind(&kind)
            .bind(from_id)
            .bind(to_id)
            .execute(&stores.pg)
            .await?;
        }
        let _ = confidence; // unused but kept in the SELECT for future heuristics
    }

    info!(
        seen = report.edges_seen,
        classified = report.edges_classified,
        directional = report.edges_directional,
        contradicts = report.edges_contradicts,
        related = report.edges_related,
        failures = report.llm_failures,
        "classify complete"
    );
    Ok(report)
}

async fn classify_pair(
    llm: &OllamaClient,
    a: &TopicSnapshot,
    b: &TopicSnapshot,
) -> Result<ClassifierOutput, KbError> {
    let prompt = build_prompt(a, b);
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
    // think=false, same trick as the tagger — full reasoning isn't needed for
    // a 4-way classification choice.
    let resp = llm
        .chat_with_options(&messages, None, Some(false))
        .await
        .map_err(|e| KbError::Other(format!("llm chat: {e}")))?;
    parse_classifier_output(&resp.content)
}

fn build_prompt(a: &TopicSnapshot, b: &TopicSnapshot) -> String {
    format!(
        "Classify the relationship between Topic A and Topic B.

Topic A: {a_label}
Sample passage from A:
\"\"\"
{a_excerpt}
\"\"\"

Topic B: {b_label}
Sample passage from B:
\"\"\"
{b_excerpt}
\"\"\"

Choose the single best relationship. CHECK THESE THREE FIRST and only fall \
back to `related` if NONE of them apply:

1. subtopic_of — one topic is a strictly more specific case or instance \
   of the other (genus / species). Examples:
     - \"ColBERTv2\" subtopic_of \"Late interaction\"
     - \"Carolingian Octagon\" subtopic_of \"Aachen Cathedral\"
     - \"centroid pruning\" subtopic_of \"Information Retrieval\"

2. builds_on — one topic directly depends on, extends, or chronologically \
   comes after the other. Use this for explicit dependencies, follow-up \
   work, or extensions. Examples:
     - \"PLAID\" builds_on \"ColBERTv2\"
     - \"Coronation of Charlemagne\" builds_on \"Donation of Pepin\"

3. contradicts — the topics represent opposing claims, approaches, or \
   views that genuinely disagree.

4. related — fallback only. Use this when the topics co-occur or share \
   themes but none of the three relationships above clearly applies.

If the relationship is directional (subtopic_of or builds_on), set \
`from` to the more specific / dependent topic. For symmetric kinds \
(related, contradicts), `from` should be null.

Output JSON only, no other text. Format:
{{\"kind\": \"<one of: subtopic_of, builds_on, contradicts, related>\", \"from\": \"A\" or \"B\" or null}}",
        a_label = a.label,
        a_excerpt = a.excerpt,
        b_label = b.label,
        b_excerpt = b.excerpt,
    )
}

fn parse_classifier_output(raw: &str) -> Result<ClassifierOutput, KbError> {
    crate::llm_parse::extract_json(raw)
}

/// Load one representative chunk per topic into memory.
///
/// Picks the highest-relevance chunk per topic via `DISTINCT ON`, returning
/// a 220-char excerpt. Cheap (one query, ~hundreds of rows) and saves doing
/// per-edge lookups in the classifier loop.
async fn load_topic_snapshots(pool: &PgPool) -> Result<HashMap<Uuid, TopicSnapshot>, KbError> {
    let rows: Vec<(Uuid, String, String)> = sqlx::query_as(
        r#"
        SELECT DISTINCT ON (tc.topic_id)
               tc.topic_id, t.label, c.content
        FROM kb_topic_chunks tc
        JOIN kb_chunks c USING (chunk_id)
        JOIN kb_topics t USING (topic_id)
        ORDER BY tc.topic_id, tc.relevance DESC
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut out = HashMap::with_capacity(rows.len());
    for (topic_id, label, content) in rows {
        let excerpt: String = content
            .chars()
            .take(REP_CHUNK_CHARS)
            .collect::<String>()
            .replace('\n', " ");
        out.insert(topic_id, TopicSnapshot { label, excerpt });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_classifier_json() {
        let raw = r#"{"kind": "subtopic_of", "from": "A"}"#;
        let out = parse_classifier_output(raw).unwrap();
        assert_eq!(out.kind, "subtopic_of");
        assert_eq!(out.from, "A");
    }

    #[test]
    fn parse_classifier_with_prefix() {
        let raw = "Sure: {\"kind\": \"related\", \"from\": \"A\"} done";
        let out = parse_classifier_output(raw).unwrap();
        assert_eq!(out.kind, "related");
    }

    #[test]
    fn parse_classifier_with_fence() {
        let raw = "```json\n{\"kind\": \"builds_on\", \"from\": \"B\"}\n```";
        let out = parse_classifier_output(raw).unwrap();
        assert_eq!(out.kind, "builds_on");
        assert_eq!(out.from, "B");
    }
}
