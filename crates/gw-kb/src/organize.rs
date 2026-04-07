//! Phase 2 organize pipeline: tag chunks, create/match topics.
//!
//! For each untagged chunk:
//!   1. Re-embed chunk content (or look up from LanceDB)
//!   2. Ask the LLM for 1-3 topic labels and named entities
//!   3. For each label: embed it, match against existing topics using
//!      `max(cosine(label_vec, topic_vec), cosine(chunk_vec, topic_vec))`
//!   4. Match → add chunk to existing topic, fold chunk vector into topic
//!      Else  → create new topic with vec = mean(label_vec, chunk_vec)
//!   5. Persist topic state and mark chunk tagged

use std::collections::HashMap;

use gw_llm::{Message, OllamaClient};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::embed::Embedder;
use crate::error::KbError;
use crate::ingest::KbStores;
use crate::topics::{
    cosine, insert_topic, insert_topic_chunk, load_all_topic_states, mark_chunk_tagged,
    slugify, update_topic_vector, TopicState,
};

/// Cosine similarity threshold for "this tag is the same as an existing topic".
///
/// Calibrated against mxbai-embed-large via Ollama on short label inputs.
/// On the validation set:
///   - exact label reuse                  → 1.00
///   - synonyms / paraphrases             → ~0.80–0.90
///   - related-but-distinct concepts      → ~0.50–0.60
///   - unrelated concepts                 → ~0.30–0.45
///
/// Note: nomic-embed-text via Ollama is unusable for this task — it produces
/// numerically identical vectors for many distinct short labels. See the
/// design discussion for the diagnosis.
pub const MATCH_THRESHOLD: f32 = 0.78;

/// Max number of existing topic labels to surface in the tagger prompt.
const PROMPT_TOPIC_BUDGET: usize = 80;

/// Max chars of chunk content to send to the tagger.
const PROMPT_CHUNK_CHARS: usize = 1800;

#[derive(Debug, Clone, Default)]
pub struct OrganizeReport {
    pub chunks_processed: usize,
    pub chunks_skipped: usize,
    pub topics_created: usize,
    pub topics_updated: usize,
    pub assignments_made: usize,
    pub llm_failures: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct OrganizeOpts {
    /// Process at most this many chunks. None = all untagged.
    pub limit: Option<usize>,
    /// Restrict to a single source.
    pub source_filter: Option<Uuid>,
    /// Re-tag chunks even if `tagged_at` is set.
    pub retag: bool,
}

impl Default for OrganizeOpts {
    fn default() -> Self {
        Self {
            limit: None,
            source_filter: None,
            retag: false,
        }
    }
}

/// Run the organize pipeline.
pub async fn organize(stores: &KbStores, opts: OrganizeOpts) -> Result<OrganizeReport, KbError> {
    let mut report = OrganizeReport::default();

    // 1. Load existing topics into memory
    let mut topic_state = load_all_topic_states(&stores.pg).await?;
    info!(loaded = topic_state.len(), "loaded existing topics into memory");

    // Used slug set to avoid collisions when creating new topics
    let mut used_slugs: std::collections::HashSet<String> =
        topic_state.iter().map(|t| t.slug.clone()).collect();

    // 2. Fetch untagged chunks
    let pending = fetch_pending_chunks(&stores.pg, &opts).await?;
    if pending.is_empty() {
        info!("no chunks to organize");
        return Ok(report);
    }
    info!(pending = pending.len(), "chunks to organize");

    // 3. Loop over chunks
    for (idx, chunk) in pending.iter().enumerate() {
        info!(
            n = idx + 1,
            total = pending.len(),
            chunk_id = %chunk.chunk_id,
            "tagging"
        );

        // 3a. Embed chunk content via sentence-transformers
        let chunk_vec = match stores.embedder.embed_one(&chunk.content) {
            Ok(v) => v,
            Err(e) => {
                warn!(err = %e, "failed to embed chunk, skipping");
                report.chunks_skipped += 1;
                continue;
            }
        };

        // 3b. Tag with LLM
        let tags = match tag_chunk(&stores.llm, chunk, &topic_state).await {
            Ok(t) => t,
            Err(e) => {
                warn!(err = %e, "tagger failed, skipping chunk");
                report.llm_failures += 1;
                report.chunks_skipped += 1;
                continue;
            }
        };
        debug!(?tags, "tagger output");

        if tags.topics.is_empty() {
            warn!(chunk_id = %chunk.chunk_id, "tagger returned no topics");
            report.chunks_skipped += 1;
            continue;
        }

        // 3c. Match each label against existing topics by pure label-to-label
        //     similarity (cosine of new label embedding vs the topic's frozen
        //     `label_vector`). Snapshot the topic list at the start of this
        //     chunk so new topics created earlier in this chunk's loop don't
        //     accidentally absorb later labels for the same chunk.
        let snapshot_len = topic_state.len();
        // Map of (topic_id -> best score) — dedupes when multiple labels
        // for THIS chunk all map to the same existing topic.
        let mut existing_assignments: HashMap<Uuid, f32> = HashMap::new();
        // New topics created during this chunk's processing, not yet inserted.
        let mut pending_new: Vec<TopicState> = Vec::new();

        for label in &tags.topics {
            let label_clean = label.trim();
            if label_clean.is_empty() {
                continue;
            }
            let label_vec = match embed_label(&stores.embedder, label_clean) {
                Ok(v) => v,
                Err(e) => {
                    warn!(err = %e, "failed to embed label, skipping");
                    continue;
                }
            };

            // Pure label_vec → topic.label_vector match against the
            // pre-chunk snapshot.
            let mut best: Option<(usize, f32)> = None;
            for (i, t) in topic_state[..snapshot_len].iter().enumerate() {
                let score = cosine(&label_vec, &t.label_vector);
                if best.map_or(true, |(_, s)| score > s) {
                    best = Some((i, score));
                }
            }

            if let Some((i, score)) = best {
                if score >= MATCH_THRESHOLD {
                    let topic_id = topic_state[i].topic_id;
                    debug!(
                        topic = %topic_state[i].label,
                        score,
                        "matched to existing topic"
                    );
                    existing_assignments
                        .entry(topic_id)
                        .and_modify(|s| {
                            if score > *s {
                                *s = score;
                            }
                        })
                        .or_insert(score);
                    continue;
                }
            }

            // No existing match. Check whether an earlier label of this
            // same chunk already produced a near-duplicate.
            let already_pending = pending_new
                .iter()
                .any(|t| cosine(&label_vec, &t.label_vector) >= MATCH_THRESHOLD);
            if already_pending {
                debug!(label = label_clean, "duplicate of pending new topic, skipping");
                continue;
            }

            // Create a new topic. label_vector is the pure label embedding.
            // vector starts as mean(label, chunk) and grows from there.
            let initial_vector = mean_two(&label_vec, &chunk_vec);
            let slug = unique_slug(&label_clean.to_lowercase(), &mut used_slugs);
            pending_new.push(TopicState {
                topic_id: Uuid::nil(),
                label: label_clean.to_string(),
                slug,
                chunk_count: 1,
                label_vector: label_vec,
                vector: initial_vector,
                dirty: false,
                created_in_run: true,
            });
        }

        // 3d. Apply existing-topic assignments
        for (topic_id, relevance) in &existing_assignments {
            let i = topic_state
                .iter()
                .position(|t| t.topic_id == *topic_id)
                .expect("topic_id from snapshot must exist in topic_state");
            topic_state[i].update_vector_with_member(&chunk_vec);
            insert_topic_chunk(&stores.pg, *topic_id, chunk.chunk_id, *relevance).await?;
            report.assignments_made += 1;
        }

        // 3e. Insert new topics from the holding buffer
        for mut new_t in pending_new {
            let id = insert_topic(
                &stores.pg,
                &new_t.label,
                &new_t.slug,
                &new_t.label_vector,
                &new_t.vector,
                new_t.chunk_count as i32,
            )
            .await?;
            new_t.topic_id = id;
            insert_topic_chunk(&stores.pg, id, chunk.chunk_id, 1.0).await?;
            report.assignments_made += 1;
            report.topics_created += 1;
            debug!(label = %new_t.label, %id, "created new topic");
            topic_state.push(new_t);
        }

        // 3d. Mark chunk tagged + entities
        mark_chunk_tagged(&stores.pg, chunk.chunk_id, &tags.entities).await?;
        report.chunks_processed += 1;
    }

    // 4. Flush dirty topic vectors back to Postgres. We flush regardless
    //    of `created_in_run` — a topic created earlier in the same run
    //    can still accumulate members that need to be persisted.
    let mut flushed = 0;
    for t in &mut topic_state {
        if t.dirty {
            update_topic_vector(&stores.pg, t.topic_id, &t.vector, t.chunk_count as i32).await?;
            t.dirty = false;
            flushed += 1;
        }
    }
    report.topics_updated = flushed;

    info!(
        processed = report.chunks_processed,
        skipped = report.chunks_skipped,
        new = report.topics_created,
        updated = report.topics_updated,
        assignments = report.assignments_made,
        "organize complete"
    );

    Ok(report)
}

#[derive(Debug)]
struct PendingChunk {
    chunk_id: Uuid,
    source_title: String,
    heading_path: Vec<String>,
    content: String,
}

async fn fetch_pending_chunks(
    pool: &PgPool,
    opts: &OrganizeOpts,
) -> Result<Vec<PendingChunk>, KbError> {
    // Build query parts. We pass an Option for source_id and a flag for retag.
    let limit = opts.limit.map(|n| n as i64).unwrap_or(i64::MAX);
    let source_filter = opts.source_filter;
    let retag = opts.retag;

    let rows: Vec<(Uuid, String, Vec<String>, String)> = sqlx::query_as(
        r#"
        SELECT c.chunk_id, s.title, c.heading_path, c.content
        FROM kb_chunks c
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE ($1 OR c.tagged_at IS NULL)
          AND ($2::uuid IS NULL OR c.source_id = $2::uuid)
        ORDER BY c.source_id, c.ordinal
        LIMIT $3
        "#,
    )
    .bind(retag)
    .bind(source_filter)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(chunk_id, title, path, content)| PendingChunk {
            chunk_id,
            source_title: title,
            heading_path: path,
            content,
        })
        .collect())
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct TaggerOutput {
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    entities: Vec<String>,
}

async fn tag_chunk(
    llm: &OllamaClient,
    chunk: &PendingChunk,
    topics: &[TopicState],
) -> Result<TaggerOutput, KbError> {
    let prompt = build_tagger_prompt(chunk, topics);
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

    // think=false is critical: with thinking enabled, qwen3.5:9b emits
    // ~5000 tokens of reasoning before the JSON, taking ~60s per call.
    // With it disabled the same call takes ~0.7s. The tagger doesn't need
    // visible reasoning — it just emits a JSON object.
    let resp = llm
        .chat_with_options(&messages, None, Some(false))
        .await
        .map_err(|e| KbError::Other(format!("llm chat: {e}")))?;
    parse_tagger_output(&resp.content)
}

const SYSTEM_PROMPT: &str = "You are a topic tagger for a knowledge base of \
research and reference documents. Output JSON only — no prose, no markdown \
fences, no commentary.";

fn build_tagger_prompt(chunk: &PendingChunk, topics: &[TopicState]) -> String {
    // Pick the most-used topics first to bias toward established labels.
    let mut sorted: Vec<&TopicState> = topics.iter().collect();
    sorted.sort_by(|a, b| b.chunk_count.cmp(&a.chunk_count));
    let sample: Vec<&str> = sorted
        .iter()
        .take(PROMPT_TOPIC_BUDGET)
        .map(|t| t.label.as_str())
        .collect();

    let existing_block = if sample.is_empty() {
        String::from("(none yet — feel free to invent suitable labels)")
    } else {
        let mut s = String::new();
        for label in &sample {
            s.push_str("- ");
            s.push_str(label);
            s.push('\n');
        }
        s
    };

    let heading = if chunk.heading_path.is_empty() {
        "(root)".to_string()
    } else {
        chunk.heading_path.join(" > ")
    };

    let content_truncated: String = chunk.content.chars().take(PROMPT_CHUNK_CHARS).collect();

    format!(
        "Tag the passage below with 1–3 specific topic labels and extract \
any named entities (people, places, organisations, models, datasets, events, works).

GUIDELINES FOR TOPIC LABELS:
- Each label is a 2–4 word noun phrase
- Prefer SPECIFIC over GENERIC. Good: \"Coronation of Charlemagne\", \
\"Carolingian Octagon\", \"Centroid Pruning\". Bad: \"Medieval History\", \
\"Catholic Church\", \"Information Retrieval\" — these are too broad.
- Pick the topics that are *distinctive* to this passage. If the passage \
discusses a particular event, technique, or named concept, use that as a label.
- Avoid emitting multiple labels that are paraphrases of each other (e.g. \
\"Medieval Popes\" + \"Papal History\" — pick one, or pick a more specific aspect).
- Reuse an existing label below if it fits exactly. Don't invent a synonym.

EXISTING TOPICS (reuse when applicable):
{existing_block}
Source: {source_title}
Section: {heading}

Passage:
\"\"\"
{content_truncated}
\"\"\"

Output JSON only, no other text:
{{\"topics\": [\"label1\", \"label2\"], \"entities\": [\"entity1\", \"entity2\"]}}",
        source_title = chunk.source_title,
    )
}

/// Parse the tagger's response. Strips markdown fences and finds the
/// outermost JSON object before deserializing.
fn parse_tagger_output(raw: &str) -> Result<TaggerOutput, KbError> {
    let stripped = strip_code_fences(raw);
    // Find the first '{' and last '}' to handle prefixes/suffixes
    let start = stripped.find('{');
    let end = stripped.rfind('}');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e >= s => &stripped[s..=e],
        _ => return Err(KbError::Other(format!("no JSON object in response: {raw:?}"))),
    };
    let parsed: TaggerOutput = serde_json::from_str(slice)
        .map_err(|e| KbError::Other(format!("json parse: {e} in {slice:?}")))?;
    Ok(TaggerOutput {
        topics: parsed
            .topics
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        entities: parsed
            .entities
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    })
}

fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        rest.trim_start_matches('\n').trim_end_matches("```").trim().to_string()
    } else if let Some(rest) = trimmed.strip_prefix("```") {
        rest.trim_start_matches('\n').trim_end_matches("```").trim().to_string()
    } else {
        trimmed.to_string()
    }
}

/// Embed a topic label via sentence-transformers. Kept as a distinct entry
/// point so we can introduce label-specific normalization (case, template,
/// prefix) in one place if needed.
fn embed_label(embedder: &Embedder, label: &str) -> Result<Vec<f32>, KbError> {
    embedder.embed_one(label)
}

fn mean_two(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect()
}

/// Pick a slug that doesn't collide with `used`. Mutates `used` to record
/// the chosen slug.
fn unique_slug(label: &str, used: &mut std::collections::HashSet<String>) -> String {
    let base = slugify(label);
    if !used.contains(&base) {
        used.insert(base.clone());
        return base;
    }
    let mut n = 2;
    loop {
        let candidate = format!("{}-{}", base, n);
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_json() {
        let raw = r#"{"topics": ["a", "b"], "entities": ["X"]}"#;
        let out = parse_tagger_output(raw).unwrap();
        assert_eq!(out.topics, vec!["a", "b"]);
        assert_eq!(out.entities, vec!["X"]);
    }

    #[test]
    fn parse_with_code_fence() {
        let raw = "```json\n{\"topics\": [\"a\"], \"entities\": []}\n```";
        let out = parse_tagger_output(raw).unwrap();
        assert_eq!(out.topics, vec!["a"]);
        assert!(out.entities.is_empty());
    }

    #[test]
    fn parse_with_prefix_text() {
        let raw = "Sure, here you go: {\"topics\": [\"a\"], \"entities\": []} cheers!";
        let out = parse_tagger_output(raw).unwrap();
        assert_eq!(out.topics, vec!["a"]);
    }

    #[test]
    fn parse_drops_empty_strings() {
        let raw = r#"{"topics": ["a", "  ", ""], "entities": [""]}"#;
        let out = parse_tagger_output(raw).unwrap();
        assert_eq!(out.topics, vec!["a"]);
        assert!(out.entities.is_empty());
    }

    #[test]
    fn unique_slug_handles_collisions() {
        let mut used = std::collections::HashSet::new();
        used.insert("foo".to_string());
        assert_eq!(unique_slug("Foo", &mut used), "foo-2");
        assert_eq!(unique_slug("foo", &mut used), "foo-3");
    }

    #[test]
    fn mean_two_basic() {
        let m = mean_two(&[1.0, 0.0], &[0.0, 1.0]);
        assert_eq!(m, vec![0.5, 0.5]);
    }
}
