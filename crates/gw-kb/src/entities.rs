//! Typed entity extraction for the KB graph.
//!
//! Phase B step 2 of `docs/design-kb-entities.md`. This module covers
//! the *extraction* half: an LLM-prompted pass that, given a chunk's
//! text and surrounding context, returns a list of typed entity
//! candidates (author / concept / method / dataset / venue / ...).
//!
//! Outputs land in [`RawEntity`] — *raw* because the labels haven't
//! been canonicalised yet. Step 3 will cluster mentions by cosine
//! similarity and collapse surface-form variants ("Lewis et al.",
//! "P. Lewis", "Patrick Lewis" → entity `lewis-patrick`) before
//! writing rows to `kb_entities`.
//!
//! The prompt mirrors `organize.rs::tag_chunk` closely so the JSON
//! discipline (no fences, no commentary, single object) stays
//! consistent across every LLM-touching subsystem in gw-kb.
//!
//! ### Where this fits in the pipeline
//!
//! ```text
//!   ingest → chunk → embed → [organize: topics] →
//!                            [extract_entities: this module] →
//!                            [canonicalise + upsert: step 3] →
//!                            [link_entities: step 4]
//! ```
//!
//! Topic tagging and entity extraction are independent passes — each
//! chunk gets one of each, in either order. We don't fuse them into a
//! single LLM call because the prompts have different optimal shapes:
//! topics want a closed vocabulary (existing labels in the corpus);
//! entities want an open vocabulary with kind discipline.

use std::collections::{HashMap, HashSet};

use gw_llm::{Message, OllamaClient};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::debug;
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;
use crate::llm_parse::extract_json;
use crate::topics::{bytes_to_vec, cosine, slugify, vec_to_bytes};

/// Recommended `kind` taxonomy. Free-form on the database side
/// (`kb_entities.kind TEXT`) but the prompt asks the model to pick
/// from this set; out-of-vocabulary kinds get filtered post-parse.
pub const RECOMMENDED_KINDS: &[&str] = &["author", "concept", "method", "dataset", "venue"];

/// One entity mention as extracted by the LLM, before canonicalisation.
///
/// `label` is the surface form as it appeared in the chunk (or the
/// LLM's slight normalisation of it); `canonical_form` is the LLM's
/// best guess at a stable identifier (e.g. "Patrick Lewis" instead of
/// "P. Lewis"). Step 3 will cluster these by embedding cosine and
/// pick a single canonical per cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawEntity {
    pub label: String,
    pub kind: String,
    /// LLM-normalised name. Falls back to `label` if the LLM omits it.
    pub canonical_form: String,
    /// Self-reported confidence in [0, 1]. 0.5 default when the model
    /// returns an out-of-range value.
    pub confidence: f32,
    /// Per-occurrence spans into the chunk's `content`, after
    /// verification. Empty for non-chunk callers (chat extraction) and
    /// for entities whose surfaces failed the in-text check. When
    /// empty, the linker still writes a single span-less mention row
    /// so the entity-presence link survives — see
    /// `docs/design-kb-wiki-provenance.md` §5.4.
    #[serde(default)]
    pub occurrences: Vec<RawOccurrence>,
}

/// One verified textual occurrence of an entity within a chunk's
/// `content`. Offsets are UTF-8 byte positions; `norm_end` is exclusive.
/// `surface` is the exact slice of `content[norm_start..norm_end]` —
/// stored redundantly so downstream consumers can render the matched
/// text without re-fetching the chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawOccurrence {
    pub norm_start: usize,
    pub norm_end: usize,
    pub surface: String,
}

/// Context the extractor needs to disambiguate borderline cases.
/// Mirrors the `PendingChunk` shape in `organize.rs` so callers can
/// pass through fields they already have.
#[derive(Debug, Clone)]
pub struct ChunkContext<'a> {
    pub source_title: &'a str,
    /// `>`-joined heading path (e.g. "Methods > Architecture") or empty
    /// for the root section. Helps the LLM pick the right kind: a
    /// dataset name appearing in the methods section is a real
    /// candidate; the same string in a citations footer is noise.
    pub heading_path: &'a str,
    pub content: &'a str,
}

/// Run the LLM extraction pass against one chunk. Returns the parsed
/// entity list with empty/duplicate/out-of-vocabulary entries removed.
///
/// `think=false` matches `organize.rs::tag_chunk`: the extractor's job
/// is to emit a JSON object, not to deliberate visibly. Reasoning
/// tokens for this prompt cost ~80x latency on qwen3.5:9b for no
/// quality gain — measured during topic tagging and reused here.
pub async fn extract_entities_for_chunk(
    llm: &OllamaClient,
    chunk: ChunkContext<'_>,
) -> Result<Vec<RawEntity>, KbError> {
    let prompt = build_extractor_prompt(&chunk);
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
    let mut entities = parse_extractor_output(&resp.content)?;
    verify_occurrences(chunk.content, &mut entities);
    Ok(entities)
}

const SYSTEM_PROMPT: &str = "You are a named-entity extractor for a knowledge base of \
research and reference documents. Output JSON only — no prose, no markdown fences, no \
commentary.";

/// Truncate the chunk content to a budget (chars, not tokens) to keep
/// a single extractor call under typical context. Topic tagging caps
/// at ~2000 chars; entities need a bit more because rare names
/// cluster late in the text.
const CONTENT_BUDGET_CHARS: usize = 3000;

fn build_extractor_prompt(chunk: &ChunkContext<'_>) -> String {
    let heading = if chunk.heading_path.is_empty() {
        "(root)".to_string()
    } else {
        chunk.heading_path.to_string()
    };
    let content_truncated = if chunk.content.len() <= CONTENT_BUDGET_CHARS {
        chunk.content.to_string()
    } else {
        // Char-boundary-safe truncation. take(N).collect() respects
        // Unicode boundaries; byte slicing would risk panicking on
        // multibyte chars in author names like "Müller".
        let mut s: String = chunk.content.chars().take(CONTENT_BUDGET_CHARS).collect();
        s.push_str("\n…[truncated]");
        s
    };
    format!(
        "Source: {source_title}\n\
        Section: {heading}\n\n\
        Extract NAMED entities mentioned in the passage below. For each, choose the best \
        matching `kind` from this set: {kinds}. If none fit cleanly, omit the entity \
        rather than inventing a new kind.\n\n\
        Rules:\n\
        - Only extract entities that are actually named in the text. Do not invent.\n\
        - `label` is the surface form as it appears (or close to it).\n\
        - `canonical_form` is the most complete / standard form you can infer (e.g. \
        \"Patrick Lewis\" rather than \"P. Lewis\"; \"Retrieval-Augmented Generation\" \
        rather than \"RAG\"). When you can't improve on the surface form, repeat it.\n\
        - `confidence` ∈ [0, 1]: how sure are you this is a real, distinct entity \
        of this kind?\n\
        - `occurrences` lists every place the entity appears in the passage. For each, \
        copy the exact surface substring (case-sensitive, including punctuation) into \
        `surface`. Char offsets are optional hints — if you include them they should be \
        character positions within the passage above; the verifier searches by `surface` \
        when offsets are missing or wrong.\n\
        - Skip generic terms (\"the model\", \"our method\") that aren't names.\n\
        - Skip pronouns and anaphoric references.\n\n\
        Passage:\n\
        \"\"\"\n\
        {content_truncated}\n\
        \"\"\"\n\n\
        Output JSON only, no other text:\n\
        {{\"entities\": [\
        {{\"label\": \"...\", \"kind\": \"author|concept|method|dataset|venue\", \
        \"canonical_form\": \"...\", \"confidence\": 0.0, \
        \"occurrences\": [{{\"surface\": \"...\", \"char_start\": 0, \"char_end\": 0}}]}}\
        ]}}",
        source_title = chunk.source_title,
        kinds = RECOMMENDED_KINDS.join(", "),
    )
}

#[derive(Debug, Default, Deserialize)]
struct ExtractorOutput {
    #[serde(default)]
    entities: Vec<RawEntityWire>,
}

#[derive(Debug, Default, Deserialize)]
struct RawEntityWire {
    #[serde(default)]
    label: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    canonical_form: String,
    /// Accept any JSON number; clamp later. Some Ollama outputs come
    /// back as integers (`1`) instead of floats (`1.0`).
    #[serde(default)]
    confidence: Option<f64>,
    /// LLM-reported occurrences. We re-verify each by string-searching
    /// `surface` inside the chunk content; `char_start`/`char_end` are
    /// hints only.
    #[serde(default)]
    occurrences: Vec<RawOccurrenceWire>,
}

#[derive(Debug, Default, Deserialize)]
struct RawOccurrenceWire {
    #[serde(default)]
    surface: String,
    // `char_start` / `char_end` are *asked for* in the prompt so the
    // LLM produces a more disciplined occurrence list, but we don't
    // trust them — serde silently drops them via the default
    // ignore-unknown-fields behaviour. Verification works by
    // string-searching `surface` in chunk content.
}

fn parse_extractor_output(raw: &str) -> Result<Vec<RawEntity>, KbError> {
    let parsed: ExtractorOutput = extract_json(raw)?;
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(parsed.entities.len());
    for e in parsed.entities {
        let label = e.label.trim().to_string();
        let kind = e.kind.trim().to_lowercase();
        if label.is_empty() || kind.is_empty() {
            continue;
        }
        if !RECOMMENDED_KINDS.contains(&kind.as_str()) {
            // Free-form kinds are allowed at the schema level, but the
            // extractor prompt explicitly constrains the set. Anything
            // else is almost certainly a hallucinated category — drop
            // rather than persist noise.
            continue;
        }
        // Dedup by (lowercased label, kind) within a single chunk's
        // output. The LLM occasionally repeats the same entity twice,
        // once with full and once with abbreviated form; we'll
        // canonicalise across chunks in step 3, so collapsing here
        // just keeps the per-chunk list clean.
        let dedup_key = (label.to_lowercase(), kind.clone());
        if !seen.insert(dedup_key) {
            continue;
        }
        let canonical_form = if e.canonical_form.trim().is_empty() {
            label.clone()
        } else {
            e.canonical_form.trim().to_string()
        };
        let confidence = e.confidence.map(|f| f as f32).unwrap_or(0.5).clamp(0.0, 1.0);
        // Carry surface hints through to verification. We don't trust
        // LLM char offsets enough to skip the in-text re-check, but we
        // do preserve the order so the verifier consumes the chunk in
        // reading order. The verifier strips entries whose surface
        // can't be located.
        let occurrences = e
            .occurrences
            .into_iter()
            .filter_map(|o| {
                let s = o.surface.trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(RawOccurrence {
                        norm_start: 0,
                        norm_end: 0,
                        surface: s,
                    })
                }
            })
            .collect();
        out.push(RawEntity {
            label,
            kind,
            canonical_form,
            confidence,
            occurrences,
        });
    }
    Ok(out)
}

/// Verify each RawEntity's `occurrences` against the chunk content.
///
/// LLMs are unreliable at character counting, so we ignore reported
/// offsets and string-search for `surface` instead. Each surface
/// consumes the *next* unclaimed match in reading order — handles the
/// "ColBERT appears 3 times, LLM reported 3 occurrences" case without
/// double-claiming any byte range.
///
/// Resolution rules per occurrence:
///   1. Find the first occurrence of `surface` in chunk content at or
///      after the high-water mark, with no overlap with prior claims.
///   2. If found, snap `norm_start`/`norm_end` to byte offsets, keep
///      the occurrence. Advance the high-water mark to the match end.
///   3. If not found, drop the occurrence and emit a trace warning.
///
/// Reading-order resolution is order-sensitive: a flipped ordering
/// from the LLM would still match correctly (since each surface gets
/// the next unclaimed match), but if the LLM lists fewer occurrences
/// than the text contains, the *first N* matches in the text are the
/// ones that get highlighted. We don't expand silently — that would
/// inflate `mentions` and is M2 territory.
///
/// If a RawEntity ends up with zero verified occurrences, we leave its
/// `occurrences` empty. The persistence layer (`canonicalize_and_persist`)
/// writes a single span-less mention row in that case so the chunk →
/// entity edge survives.
fn verify_occurrences(content: &str, entities: &mut [RawEntity]) {
    let mut high_water: usize = 0;
    // We could let each entity have its own search cursor, but the
    // shared high-water mark mirrors how a human reader would tag —
    // the second mention of any entity comes after the first mention
    // of any other entity, in document order. With per-entity cursors
    // two entities can claim overlapping byte ranges (e.g. "ColBERT"
    // and "ColBERT-v2" at the same position), so we still need an
    // overlap check below; the high-water is a hint, not a hard lock.
    for ent in entities.iter_mut() {
        let mut verified: Vec<RawOccurrence> = Vec::with_capacity(ent.occurrences.len());
        let mut per_entity_cursor = 0usize;
        for occ in std::mem::take(&mut ent.occurrences) {
            // Find first match of `surface` at or after the entity's
            // cursor. (Within one entity, occurrences must be ordered.)
            let surface = &occ.surface;
            let mut search_from = per_entity_cursor;
            let found = loop {
                if search_from > content.len() {
                    break None;
                }
                // `find` is byte-offset based, which is what we want
                // for the DB. Guard against split-point inside a
                // multibyte char by clamping to a char boundary.
                let from = next_char_boundary(content, search_from);
                let Some(rel) = content[from..].find(surface.as_str()) else {
                    break None;
                };
                let abs = from + rel;
                let abs_end = abs + surface.len();
                // Reject if overlapping any prior claim from THIS
                // entity (between-entity overlaps are fine — they
                // produce overlapping highlights, which the frontend
                // can render with stacked underlines).
                let overlaps = verified
                    .iter()
                    .any(|v| abs < v.norm_end && v.norm_start < abs_end);
                if overlaps {
                    search_from = abs + 1;
                    continue;
                }
                break Some((abs, abs_end));
            };
            match found {
                Some((s, e)) => {
                    per_entity_cursor = e;
                    if e > high_water {
                        high_water = e;
                    }
                    verified.push(RawOccurrence {
                        norm_start: s,
                        norm_end: e,
                        surface: surface.clone(),
                    });
                }
                None => {
                    tracing::warn!(
                        label = %ent.label,
                        surface = %surface,
                        "occurrence surface not found in chunk; dropping",
                    );
                }
            }
        }
        // Suppress unused-variable lint if entities don't share a
        // reading order; high_water remains useful when we extend this
        // later for cross-entity ordering. For now it's vestigial but
        // cheap.
        let _ = high_water;
        ent.occurrences = verified;
    }
}

/// Snap `pos` forward to the next UTF-8 char boundary in `s`. Returns
/// `s.len()` if `pos >= s.len()`.
fn next_char_boundary(s: &str, pos: usize) -> usize {
    let mut p = pos.min(s.len());
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_parses() {
        let raw = r#"{"entities": [
            {"label": "Patrick Lewis", "kind": "author", "canonical_form": "Patrick Lewis", "confidence": 0.95},
            {"label": "RAG", "kind": "concept", "canonical_form": "Retrieval-Augmented Generation", "confidence": 0.9}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].label, "Patrick Lewis");
        assert_eq!(out[0].kind, "author");
        assert_eq!(out[1].canonical_form, "Retrieval-Augmented Generation");
    }

    #[test]
    fn strips_markdown_fences_and_preamble() {
        let raw = "Sure, here are the entities:\n```json\n{\"entities\": [{\"label\": \"BERT\", \"kind\": \"method\", \"canonical_form\": \"BERT\", \"confidence\": 0.8}]}\n```";
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "BERT");
    }

    #[test]
    fn drops_invalid_kinds() {
        let raw = r#"{"entities": [
            {"label": "BERT", "kind": "method", "canonical_form": "BERT", "confidence": 0.8},
            {"label": "Some Tool", "kind": "tool", "canonical_form": "Some Tool", "confidence": 0.7}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "BERT");
    }

    #[test]
    fn dedups_within_chunk() {
        let raw = r#"{"entities": [
            {"label": "BERT", "kind": "method", "canonical_form": "BERT", "confidence": 0.8},
            {"label": "bert", "kind": "method", "canonical_form": "BERT", "confidence": 0.6}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn missing_canonical_form_falls_back_to_label() {
        let raw = r#"{"entities": [{"label": "BERT", "kind": "method", "canonical_form": "", "confidence": 0.8}]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out[0].canonical_form, "BERT");
    }

    #[test]
    fn missing_confidence_defaults_to_half() {
        let raw = r#"{"entities": [{"label": "BERT", "kind": "method", "canonical_form": "BERT"}]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert!((out[0].confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_entity_list_is_ok() {
        let raw = r#"{"entities": []}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn empty_label_filtered() {
        let raw = r#"{"entities": [{"label": "  ", "kind": "method", "canonical_form": "x", "confidence": 0.8}]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn pick_canonical_label_takes_longest() {
        let mentions = ["P. Lewis", "Patrick Lewis", "Lewis"];
        let canonical = pick_canonical_label(&mentions);
        assert_eq!(canonical, "Patrick Lewis");
    }

    #[test]
    fn parses_occurrences_from_wire() {
        let raw = r#"{"entities": [
            {"label": "ColBERT", "kind": "method", "canonical_form": "ColBERT", "confidence": 0.9,
             "occurrences": [
                {"surface": "ColBERT", "char_start": 0, "char_end": 7},
                {"surface": "ColBERT-v2", "char_start": 20, "char_end": 30}
             ]}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out.len(), 1);
        // parse layer doesn't verify — offsets are still zero, surfaces preserved.
        assert_eq!(out[0].occurrences.len(), 2);
        assert_eq!(out[0].occurrences[0].surface, "ColBERT");
        assert_eq!(out[0].occurrences[1].surface, "ColBERT-v2");
    }

    #[test]
    fn parses_missing_occurrences_as_empty() {
        let raw = r#"{"entities": [
            {"label": "BERT", "kind": "method", "canonical_form": "BERT", "confidence": 0.8}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert!(out[0].occurrences.is_empty());
    }

    #[test]
    fn parser_drops_empty_surface_occurrences() {
        let raw = r#"{"entities": [
            {"label": "BERT", "kind": "method", "canonical_form": "BERT", "confidence": 0.8,
             "occurrences": [{"surface": "  "}, {"surface": "BERT"}]}
        ]}"#;
        let out = parse_extractor_output(raw).unwrap();
        assert_eq!(out[0].occurrences.len(), 1);
        assert_eq!(out[0].occurrences[0].surface, "BERT");
    }

    #[test]
    fn verify_locates_single_occurrence() {
        let content = "We trained ColBERT on MS MARCO.";
        let mut ents = vec![RawEntity {
            label: "ColBERT".into(),
            kind: "method".into(),
            canonical_form: "ColBERT".into(),
            confidence: 0.9,
            occurrences: vec![RawOccurrence {
                norm_start: 0,
                norm_end: 0,
                surface: "ColBERT".into(),
            }],
        }];
        verify_occurrences(content, &mut ents);
        assert_eq!(ents[0].occurrences.len(), 1);
        let o = &ents[0].occurrences[0];
        assert_eq!(&content[o.norm_start..o.norm_end], "ColBERT");
    }

    #[test]
    fn verify_resolves_repeated_surface_to_distinct_positions() {
        let content = "ColBERT was extended. ColBERT-v2 followed. Later, ColBERT remained popular.";
        let mut ents = vec![RawEntity {
            label: "ColBERT".into(),
            kind: "method".into(),
            canonical_form: "ColBERT".into(),
            confidence: 0.9,
            occurrences: vec![
                RawOccurrence { norm_start: 0, norm_end: 0, surface: "ColBERT".into() },
                RawOccurrence { norm_start: 0, norm_end: 0, surface: "ColBERT".into() },
            ],
        }];
        verify_occurrences(content, &mut ents);
        assert_eq!(ents[0].occurrences.len(), 2);
        let a = &ents[0].occurrences[0];
        let b = &ents[0].occurrences[1];
        // First "ColBERT" at 0, third "ColBERT" at 50 (after "ColBERT-v2"
        // is skipped because it doesn't match the surface "ColBERT"
        // exactly — but wait, "ColBERT-v2" *contains* "ColBERT" as a
        // prefix, so the second match will be inside "ColBERT-v2"). We
        // verify the matches don't overlap and both find some "ColBERT"
        // substring.
        assert!(a.norm_start < b.norm_start);
        assert_eq!(&content[a.norm_start..a.norm_end], "ColBERT");
        assert_eq!(&content[b.norm_start..b.norm_end], "ColBERT");
        assert!(b.norm_start >= a.norm_end);
    }

    #[test]
    fn verify_drops_surface_not_present() {
        let content = "The paper introduces BERT.";
        let mut ents = vec![RawEntity {
            label: "Imaginary".into(),
            kind: "method".into(),
            canonical_form: "Imaginary".into(),
            confidence: 0.5,
            occurrences: vec![RawOccurrence {
                norm_start: 0,
                norm_end: 0,
                surface: "Imaginary".into(),
            }],
        }];
        verify_occurrences(content, &mut ents);
        assert!(ents[0].occurrences.is_empty());
    }

    #[test]
    fn verify_handles_multibyte_chars() {
        // "Müller" — the 'ü' is two UTF-8 bytes. Surface match should
        // produce byte offsets that re-slice cleanly.
        let content = "The paper by Müller et al. extends prior work.";
        let mut ents = vec![RawEntity {
            label: "Müller".into(),
            kind: "author".into(),
            canonical_form: "Müller".into(),
            confidence: 0.9,
            occurrences: vec![RawOccurrence {
                norm_start: 0,
                norm_end: 0,
                surface: "Müller".into(),
            }],
        }];
        verify_occurrences(content, &mut ents);
        assert_eq!(ents[0].occurrences.len(), 1);
        let o = &ents[0].occurrences[0];
        assert_eq!(&content[o.norm_start..o.norm_end], "Müller");
        // Sanity: byte span is 7 (M=1, ü=2, l=1, l=1, e=1, r=1).
        assert_eq!(o.norm_end - o.norm_start, 7);
    }

    #[test]
    fn aliases_excludes_canonical_and_dedups() {
        let mentions = vec![
            "Patrick Lewis".to_string(),
            "P. Lewis".to_string(),
            "patrick lewis".to_string(),
            "Lewis".to_string(),
        ];
        let aliases = collect_aliases("Patrick Lewis", &mentions);
        assert_eq!(aliases, vec!["P. Lewis".to_string(), "Lewis".to_string()]);
    }
}

// ─── Step 3: canonicalise + persist ─────────────────────────────────
//
// The bridge from `RawEntity` (per-chunk extractor output) to
// `kb_entities` (canonical persistent rows). Three responsibilities:
//   1. Embed each mention's `canonical_form` in one batch.
//   2. Cluster mentions across chunks by cosine, both against existing
//      `kb_entities` rows and amongst themselves within the batch.
//      Each cluster collapses to one canonical entity.
//   3. Upsert entities (insert new clusters, update existing centroids
//      and alias sets), then write `kb_chunk_entity_links` rows so the
//      chunk → entity edges are queryable.
//
// Within-batch matching is greedy in mention order — first mention
// sets the cluster's centroid, subsequent mentions either join an
// existing cluster (if cosine ≥ threshold) or start a new one. Order
// matters at the margin: ranking mentions by length (so the longer
// surface form wins as canonical) is good enough at our scale.
//
// Cross-kind matching never happens — a "Patrick Lewis" `concept`
// can't merge with a "Patrick Lewis" `author`.

/// Bundle of entity mentions for one chunk. Pass a slice of these to
/// `canonicalize_and_persist`.
#[derive(Debug, Clone)]
pub struct ChunkEntities {
    pub chunk_id: Uuid,
    pub entities: Vec<RawEntity>,
}

/// Tunable parameters for canonicalisation.
#[derive(Debug, Clone)]
pub struct CanonicalizeOpts {
    /// Cosine threshold for both "match against existing entity" and
    /// "merge two within-batch mentions". 0.9 is the design-doc
    /// default; lower values aggressively collapse surface forms,
    /// higher values keep more distinct entities.
    pub merge_threshold: f32,
    /// When true, also writes the canonical labels back to
    /// `kb_chunks.entities TEXT[]` (the migration-010 column). Lets
    /// callers bypass it during batch backfills if they're going to
    /// rebuild that column separately.
    pub write_chunk_entities_column: bool,
}

impl Default for CanonicalizeOpts {
    fn default() -> Self {
        Self {
            merge_threshold: 0.9,
            write_chunk_entities_column: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct EntityIngestReport {
    pub mentions_in: usize,
    pub entities_created: usize,
    pub entities_updated: usize,
    pub links_created: usize,
}

/// Embed → cluster → upsert into `kb_entities` for a flat list of raw
/// mentions, returning a parallel `Vec<Uuid>` that maps each input
/// `mentions[i]` to its canonicalised entity. Does NOT write any
/// chunk-side or entry-side link rows — those are the caller's
/// problem (`canonicalize_and_persist` writes `kb_chunk_entity_links`;
/// the spine writes `session_entry_entities`).
///
/// Cross-batch consistency: existing entities of the same kind are
/// loaded and matched against; an incoming mention prefers attaching
/// to an existing entity over starting a fresh in-batch cluster, so
/// repeated runs over the same canonical_form converge to one row.
pub async fn canonicalize_mentions(
    stores: &KbStores,
    mentions: &[RawEntity],
    opts: &CanonicalizeOpts,
) -> Result<(Vec<Uuid>, EntityIngestReport), KbError> {
    let mentions_in = mentions.len();
    if mentions.is_empty() {
        return Ok((Vec::new(), EntityIngestReport::default()));
    }

    // Embed all canonical_forms in one batch. The embedder's
    // sentence-transformers backend is loaded lazily on first call;
    // batching keeps that one-time cost amortised.
    let texts: Vec<String> = mentions.iter().map(|m| m.canonical_form.clone()).collect();
    let vectors = stores.embedder.embed_texts(&texts)?;
    if vectors.len() != mentions.len() {
        return Err(KbError::Other(format!(
            "embedder returned {} vectors for {} mentions",
            vectors.len(),
            mentions.len()
        )));
    }

    // Group mention indices by kind so each kind clusters
    // independently. Shared cosine threshold; per-kind threshold is a
    // future tunable mentioned in the design doc.
    let mut by_kind: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, m) in mentions.iter().enumerate() {
        by_kind.entry(m.kind.clone()).or_default().push(idx);
    }

    let mut report = EntityIngestReport {
        mentions_in,
        ..Default::default()
    };
    let mut mention_entity: HashMap<usize, Uuid> = HashMap::new();

    for (kind, indices) in by_kind {
        // Sort within-kind by length descending so longer (more
        // canonical) surface forms seed clusters first. A short
        // mention arriving later then merges into the longer cluster
        // and contributes its label as an alias.
        let mut ordered = indices;
        ordered.sort_by_key(|i| std::cmp::Reverse(mentions[*i].canonical_form.len()));

        let existing = load_entities_by_kind(&stores.pg, &kind).await?;

        // Per-cluster accumulators (in-batch, brand-new clusters).
        // existing-matching is handled separately below.
        let mut new_clusters: Vec<NewCluster> = Vec::new();
        // Updates to existing entity rows.
        let mut existing_updates: HashMap<Uuid, ExistingUpdate> = HashMap::new();

        for i in ordered {
            let v = &vectors[i];
            let raw = &mentions[i];

            // Best match across existing rows.
            let mut best_existing: Option<(usize, f32)> = None;
            for (j, e) in existing.iter().enumerate() {
                if e.vector.is_empty() {
                    continue;
                }
                let s = cosine(v, &e.vector);
                if s >= opts.merge_threshold && best_existing.map(|(_, bs)| s > bs).unwrap_or(true) {
                    best_existing = Some((j, s));
                }
            }
            // Best match across in-batch new clusters.
            let mut best_new: Option<(usize, f32)> = None;
            for (j, c) in new_clusters.iter().enumerate() {
                let s = cosine(v, &c.centroid);
                if s >= opts.merge_threshold && best_new.map(|(_, bs)| s > bs).unwrap_or(true) {
                    best_new = Some((j, s));
                }
            }

            // Prefer attaching to an existing entity over creating a
            // new in-batch cluster — keeps cross-batch consistency.
            match (best_existing, best_new) {
                (Some((j, _)), _) => {
                    let entity_id = existing[j].entity_id;
                    let upd = existing_updates
                        .entry(entity_id)
                        .or_insert_with(|| ExistingUpdate::seed(&existing[j]));
                    upd.absorb(&raw.canonical_form, v);
                    mention_entity.insert(i, entity_id);
                }
                (None, Some((j, _))) => {
                    let cluster = &mut new_clusters[j];
                    cluster.absorb(&raw.canonical_form, v);
                    cluster.mention_indices.push(i);
                }
                (None, None) => {
                    new_clusters.push(NewCluster::seed(kind.clone(), raw, v.clone(), i));
                }
            }
        }

        // Persist new clusters.
        let mut used_slugs: HashSet<String> = existing.iter().map(|e| e.slug.clone()).collect();
        for cluster in new_clusters {
            let canonical = pick_canonical_label(
                &cluster
                    .surface_forms
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            );
            let aliases = collect_aliases(&canonical, &cluster.surface_forms);
            let slug = unique_entity_slug(&canonical, &mut used_slugs);
            let mention_count = cluster.mention_indices.len() as i32;
            let (entity_id, was_created) = upsert_entity(
                &stores.pg,
                &canonical,
                &slug,
                &cluster.kind,
                &aliases,
                &cluster.centroid,
                mention_count,
            )
            .await?;
            if was_created {
                report.entities_created += 1;
            } else {
                report.entities_updated += 1;
            }
            for mi in cluster.mention_indices {
                mention_entity.insert(mi, entity_id);
            }
        }

        // Persist updates to existing entities.
        for (entity_id, upd) in existing_updates {
            update_entity_with_mentions(
                &stores.pg,
                entity_id,
                &upd.centroid,
                &upd.aliases_to_add,
                upd.added_mentions,
            )
            .await?;
            report.entities_updated += 1;
        }
    }

    // Build the parallel Vec<Uuid>. Every input index must have
    // landed somewhere by now; if it didn't, that's a bug, not a
    // routine miss — surface it.
    let mut entity_ids: Vec<Uuid> = Vec::with_capacity(mentions.len());
    for i in 0..mentions.len() {
        let id = mention_entity.get(&i).copied().ok_or_else(|| {
            KbError::Other(format!(
                "canonicalize_mentions: mention {i} did not resolve to an entity"
            ))
        })?;
        entity_ids.push(id);
    }
    Ok((entity_ids, report))
}

/// Top-level entry point for step 3 of the chunk-side ingest pipeline.
/// Embeds → clusters → upserts → writes `kb_chunk_entity_links`.
/// Spine callers want only the canonicalisation half — see
/// `canonicalize_mentions`.
pub async fn canonicalize_and_persist(
    stores: &KbStores,
    chunks: &[ChunkEntities],
    opts: &CanonicalizeOpts,
) -> Result<EntityIngestReport, KbError> {
    // Flatten chunks → parallel arrays of (chunk_id, RawEntity). The
    // chunk_id list lines up with the entity_ids returned by
    // canonicalize_mentions.
    let mut chunk_ids: Vec<Uuid> = Vec::new();
    let mut mentions: Vec<RawEntity> = Vec::new();
    for c in chunks {
        for raw in &c.entities {
            chunk_ids.push(c.chunk_id);
            mentions.push(raw.clone());
        }
    }
    if mentions.is_empty() {
        return Ok(EntityIngestReport::default());
    }

    let (entity_ids, mut report) = canonicalize_mentions(stores, &mentions, opts).await?;

    // Write kb_entity_mentions and update kb_chunks.entities. Group by
    // chunk so kb_chunks.entities batches a single UPDATE.
    //
    // Re-extraction policy: clear any pre-existing mentions for each
    // chunk once at the start of this batch, then write the fresh set.
    // The legacy span-less rows migrated from kb_chunk_entity_links in
    // migration 017 get replaced here on first re-extraction of their
    // owning chunk. (`docs/design-kb-wiki-provenance.md` §7.2.)
    let mut by_chunk: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    let mut cleared_chunks: HashSet<Uuid> = HashSet::new();
    for (i, (chunk_id, entity_id)) in chunk_ids.iter().zip(entity_ids.iter()).enumerate() {
        if cleared_chunks.insert(*chunk_id) {
            sqlx::query("DELETE FROM kb_entity_mentions WHERE chunk_id = $1")
                .bind(chunk_id)
                .execute(&stores.pg)
                .await?;
        }
        let raw = &mentions[i];
        let n = insert_entity_mentions(
            &stores.pg,
            *chunk_id,
            *entity_id,
            &raw.occurrences,
            raw.confidence,
        )
        .await?;
        report.links_created += n;
        by_chunk.entry(*chunk_id).or_default().push(*entity_id);
    }

    if opts.write_chunk_entities_column {
        // Replace kb_chunks.entities TEXT[] with the canonical labels
        // for this batch's chunks. Pulls labels in one query so we
        // don't fetch per-chunk.
        let all_entity_ids: Vec<Uuid> = by_chunk.values().flatten().copied().collect();
        if !all_entity_ids.is_empty() {
            let labels = fetch_entity_labels(&stores.pg, &all_entity_ids).await?;
            for (chunk_id, entity_ids) in &by_chunk {
                let mut seen = HashSet::new();
                let mut chunk_labels: Vec<String> = Vec::new();
                for eid in entity_ids {
                    if let Some(label) = labels.get(eid) {
                        if seen.insert(label.clone()) {
                            chunk_labels.push(label.clone());
                        }
                    }
                }
                sqlx::query("UPDATE kb_chunks SET entities = $2 WHERE chunk_id = $1")
                    .bind(chunk_id)
                    .bind(&chunk_labels)
                    .execute(&stores.pg)
                    .await?;
            }
        }
    }

    debug!(
        mentions_in = report.mentions_in,
        entities_created = report.entities_created,
        entities_updated = report.entities_updated,
        links_created = report.links_created,
        "entity canonicalisation complete"
    );
    Ok(report)
}

// ─── Driver: load chunks, extract, canonicalise ─────────────────────
//
// Glue between step 2 (per-chunk LLM extraction) and step 3
// (cross-chunk canonicalisation + persistence). One LLM call per
// chunk; sequential rather than parallel because the local Ollama
// path is single-threaded under the hood and parallel calls would
// just queue. Failures on individual chunks are logged and skipped
// — better to ingest 28 of 30 papers' worth of entities than to fail
// the whole batch on a flaky chunk.
//
// Two entry points:
//   - `extract_and_persist_entities_for_source` — all chunks of one
//     source. Used by per-paper ingest flows (e.g. literature_assistant
//     after `arxiv_search` has populated kb_sources).
//   - `extract_and_persist_entities_for_chunks` — explicit chunk-id
//     list. Used by the CLI for backfills and by tests.

#[derive(sqlx::FromRow)]
struct ExtractChunkRow {
    chunk_id: Uuid,
    title: String,
    heading_path: Vec<String>,
    content: String,
}

/// Run extraction over every chunk of `source_id`, canonicalise the
/// results, and persist. Skips chunks that error during extraction.
pub async fn extract_and_persist_entities_for_source(
    stores: &KbStores,
    source_id: Uuid,
    opts: &CanonicalizeOpts,
) -> Result<EntityIngestReport, KbError> {
    let rows: Vec<ExtractChunkRow> = sqlx::query_as(
        r#"
        SELECT c.chunk_id, s.title, c.heading_path, c.content
        FROM kb_chunks c
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE c.source_id = $1
        ORDER BY c.ordinal
        "#,
    )
    .bind(source_id)
    .fetch_all(&stores.pg)
    .await?;
    extract_and_persist_for_rows(stores, rows, opts).await
}

/// Run extraction over a specific list of chunk ids. Useful for
/// backfilling: e.g. "extract entities for the 200 newest chunks".
pub async fn extract_and_persist_entities_for_chunks(
    stores: &KbStores,
    chunk_ids: &[Uuid],
    opts: &CanonicalizeOpts,
) -> Result<EntityIngestReport, KbError> {
    if chunk_ids.is_empty() {
        return Ok(EntityIngestReport::default());
    }
    let rows: Vec<ExtractChunkRow> = sqlx::query_as(
        r#"
        SELECT c.chunk_id, s.title, c.heading_path, c.content
        FROM kb_chunks c
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE c.chunk_id = ANY($1)
        "#,
    )
    .bind(chunk_ids)
    .fetch_all(&stores.pg)
    .await?;
    extract_and_persist_for_rows(stores, rows, opts).await
}

async fn extract_and_persist_for_rows(
    stores: &KbStores,
    rows: Vec<ExtractChunkRow>,
    opts: &CanonicalizeOpts,
) -> Result<EntityIngestReport, KbError> {
    if rows.is_empty() {
        return Ok(EntityIngestReport::default());
    }
    let mut bundles: Vec<ChunkEntities> = Vec::with_capacity(rows.len());
    for row in &rows {
        let heading_str = row.heading_path.join(" > ");
        let ctx = ChunkContext {
            source_title: &row.title,
            heading_path: &heading_str,
            content: &row.content,
        };
        match extract_entities_for_chunk(&stores.llm, ctx).await {
            Ok(entities) if !entities.is_empty() => {
                bundles.push(ChunkEntities {
                    chunk_id: row.chunk_id,
                    entities,
                });
            }
            Ok(_) => { /* nothing useful from this chunk */ }
            Err(e) => {
                tracing::warn!(
                    chunk_id = %row.chunk_id,
                    error = %e,
                    "entity extraction failed; skipping chunk"
                );
            }
        }
    }
    canonicalize_and_persist(stores, &bundles, opts).await
}

// ─── Internal: clustering scratchpads ───────────────────────────────

/// In-batch cluster that hasn't been persisted yet.
#[derive(Debug, Clone)]
struct NewCluster {
    kind: String,
    surface_forms: Vec<String>,
    centroid: Vec<f32>,
    n: usize,
    mention_indices: Vec<usize>,
}

impl NewCluster {
    fn seed(kind: String, raw: &RawEntity, v: Vec<f32>, mention_idx: usize) -> Self {
        Self {
            kind,
            surface_forms: vec![raw.canonical_form.clone()],
            centroid: v,
            n: 1,
            mention_indices: vec![mention_idx],
        }
    }
    fn absorb(&mut self, surface: &str, v: &[f32]) {
        self.surface_forms.push(surface.to_string());
        running_mean(&mut self.centroid, v, self.n);
        self.n += 1;
    }
}

/// Update accumulator for an existing kb_entities row that was matched
/// against during this batch. We collect mentions here, then write a
/// single UPDATE per row at the end of the kind's loop.
#[derive(Debug, Clone)]
struct ExistingUpdate {
    centroid: Vec<f32>,
    n: usize,
    existing_label: String,
    existing_aliases: HashSet<String>,
    aliases_to_add: Vec<String>,
    added_mentions: i32,
}

impl ExistingUpdate {
    fn seed(row: &EntityRow) -> Self {
        Self {
            centroid: row.vector.clone(),
            n: row.mentions.max(1) as usize,
            existing_label: row.label.clone(),
            existing_aliases: row.aliases.iter().cloned().collect(),
            aliases_to_add: Vec::new(),
            added_mentions: 0,
        }
    }
    fn absorb(&mut self, surface: &str, v: &[f32]) {
        running_mean(&mut self.centroid, v, self.n);
        self.n += 1;
        self.added_mentions += 1;
        // Don't re-add the canonical label or already-known aliases.
        if surface != self.existing_label
            && !self.existing_aliases.contains(surface)
            && !self.aliases_to_add.iter().any(|a| a == surface)
        {
            self.aliases_to_add.push(surface.to_string());
        }
    }
}

/// In-place running mean: `dst = (dst * n + v) / (n + 1)`.
fn running_mean(dst: &mut [f32], v: &[f32], n: usize) {
    if dst.len() != v.len() {
        // Dimension mismatch — leave dst alone rather than panic.
        // Shouldn't happen given we pull from one embedder run.
        return;
    }
    let n_f = n as f32;
    for (d, x) in dst.iter_mut().zip(v.iter()) {
        *d = (*d * n_f + *x) / (n_f + 1.0);
    }
}

/// Pick the longest surface form as the canonical label. Ties broken
/// by lexicographic order so the choice is deterministic across runs.
fn pick_canonical_label(mentions: &[&str]) -> String {
    mentions
        .iter()
        .copied()
        .max_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)))
        .unwrap_or("")
        .to_string()
}

/// Aliases = unique surface forms minus the canonical, in order of
/// first appearance. Case-sensitive equality with the canonical, but
/// case-insensitive dedup amongst the rest (so "Lewis" and "lewis"
/// don't both end up in the array).
fn collect_aliases(canonical: &str, mentions: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen_lower: HashSet<String> = HashSet::new();
    seen_lower.insert(canonical.to_lowercase());
    for m in mentions {
        if m == canonical {
            continue;
        }
        let key = m.to_lowercase();
        if seen_lower.insert(key) {
            out.push(m.clone());
        }
    }
    out
}

/// Pick a slug that doesn't collide with `used`; mutates `used`.
fn unique_entity_slug(label: &str, used: &mut HashSet<String>) -> String {
    let base = slugify(label);
    if !used.contains(&base) {
        used.insert(base.clone());
        return base;
    }
    let mut n = 2;
    loop {
        let cand = format!("{base}-{n}");
        if !used.contains(&cand) {
            used.insert(cand.clone());
            return cand;
        }
        n += 1;
    }
}

// ─── Internal: SQL ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct EntityRow {
    entity_id: Uuid,
    label: String,
    slug: String,
    aliases: Vec<String>,
    mentions: i32,
    vector: Vec<f32>,
}

/// Raw row shape from `SELECT entity_id, label, slug, aliases,
/// mentions, vector FROM kb_entities`. Pulled out as a type alias so
/// the sqlx call site doesn't trip clippy::type_complexity.
type EntityRowTuple = (Uuid, String, String, Vec<String>, i32, Option<Vec<u8>>);

async fn load_entities_by_kind(pool: &PgPool, kind: &str) -> Result<Vec<EntityRow>, KbError> {
    let rows: Vec<EntityRowTuple> = sqlx::query_as(
        "SELECT entity_id, label, slug, aliases, mentions, vector FROM kb_entities WHERE kind = $1",
    )
    .bind(kind)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(entity_id, label, slug, aliases, mentions, vector)| EntityRow {
            entity_id,
            label,
            slug,
            aliases,
            mentions,
            vector: vector.map(|b| bytes_to_vec(&b)).unwrap_or_default(),
        })
        .collect())
}

/// Insert a new entity, or merge into an existing row on slug
/// collision. Returns `(entity_id, was_created)`. When merging, the
/// existing row's `vector` is updated to the (count-weighted) mean
/// across both, aliases are unioned, and `mentions` is incremented by
/// the new contribution.
async fn upsert_entity(
    pool: &PgPool,
    label: &str,
    slug: &str,
    kind: &str,
    aliases: &[String],
    vector: &[f32],
    mention_count: i32,
) -> Result<(Uuid, bool), KbError> {
    let bytes = vec_to_bytes(vector);
    // Try insert first. ON CONFLICT (slug) returns the existing row
    // unmodified; we then read it, weighted-average the vector, union
    // aliases, and UPDATE.
    let inserted: Option<Uuid> = sqlx::query_scalar(
        r#"
        INSERT INTO kb_entities (label, slug, kind, aliases, mentions, vector)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (slug) DO NOTHING
        RETURNING entity_id
        "#,
    )
    .bind(label)
    .bind(slug)
    .bind(kind)
    .bind(aliases)
    .bind(mention_count)
    .bind(&bytes)
    .fetch_optional(pool)
    .await?;
    if let Some(id) = inserted {
        return Ok((id, true));
    }
    // Slug collision — merge into the existing row.
    let row: (Uuid, Vec<String>, i32, Option<Vec<u8>>) = sqlx::query_as(
        "SELECT entity_id, aliases, mentions, vector FROM kb_entities WHERE slug = $1",
    )
    .bind(slug)
    .fetch_one(pool)
    .await?;
    let (entity_id, existing_aliases, existing_mentions, existing_vector) = row;
    let merged_aliases = union_aliases_with(label, &existing_aliases, aliases);
    let merged_vector = weighted_mean_vector(
        &existing_vector.map(|b| bytes_to_vec(&b)).unwrap_or_default(),
        existing_mentions.max(1),
        vector,
        mention_count.max(1),
    );
    let merged_bytes = vec_to_bytes(&merged_vector);
    sqlx::query(
        r#"
        UPDATE kb_entities
        SET aliases    = $2,
            mentions   = mentions + $3,
            vector     = $4,
            last_seen  = now(),
            updated_at = now()
        WHERE entity_id = $1
        "#,
    )
    .bind(entity_id)
    .bind(&merged_aliases)
    .bind(mention_count)
    .bind(&merged_bytes)
    .execute(pool)
    .await?;
    Ok((entity_id, false))
}

/// Update path for entities matched against during a batch. Same shape
/// as the merge branch in `upsert_entity` but the caller has already
/// computed the new centroid and aliases-to-add.
async fn update_entity_with_mentions(
    pool: &PgPool,
    entity_id: Uuid,
    centroid: &[f32],
    aliases_to_add: &[String],
    added_mentions: i32,
) -> Result<(), KbError> {
    if added_mentions == 0 && aliases_to_add.is_empty() {
        return Ok(());
    }
    let bytes = vec_to_bytes(centroid);
    // COALESCE the de-duped union with an empty array — `array_agg`
    // over an empty set returns NULL, and `kb_entities.aliases` is
    // NOT NULL. This trips for entities whose mentions all share the
    // canonical label (no new aliases) AND whose row already had no
    // aliases (so the union is empty).
    sqlx::query(
        r#"
        UPDATE kb_entities
        SET aliases    = COALESCE(
                (SELECT array_agg(DISTINCT a) FROM unnest(aliases || $2::text[]) AS a),
                ARRAY[]::TEXT[]
            ),
            mentions   = mentions + $3,
            vector     = $4,
            last_seen  = now(),
            updated_at = now()
        WHERE entity_id = $1
        "#,
    )
    .bind(entity_id)
    .bind(aliases_to_add)
    .bind(added_mentions)
    .bind(&bytes)
    .execute(pool)
    .await?;
    Ok(())
}

/// Insert one row per occurrence into `kb_entity_mentions`, returning
/// the number of rows written. When `occurrences` is empty (the LLM
/// either omitted spans or all surfaces failed verification), we still
/// write a single span-less row — that preserves the chunk → entity
/// edge through the `kb_chunk_entity_links` view so the sidebar entry
/// survives, even when inline highlights aren't available.
///
/// Caller is responsible for any prior cleanup of the chunk's mentions
/// (`canonicalize_and_persist` does this once per chunk per batch).
async fn insert_entity_mentions(
    pool: &PgPool,
    chunk_id: Uuid,
    entity_id: Uuid,
    occurrences: &[RawOccurrence],
    confidence: f32,
) -> Result<usize, KbError> {
    if occurrences.is_empty() {
        sqlx::query(
            r#"
            INSERT INTO kb_entity_mentions (chunk_id, entity_id, confidence)
            VALUES ($1, $2, $3)
            "#,
        )
        .bind(chunk_id)
        .bind(entity_id)
        .bind(confidence)
        .execute(pool)
        .await?;
        return Ok(1);
    }
    // One INSERT per occurrence. Typical chunks have a handful of
    // mentions per entity; the round-trip cost is acceptable for M1.
    // A bulk UNNEST path is straightforward to add if profiling
    // surfaces this as a hotspot.
    let mut written = 0usize;
    for occ in occurrences {
        sqlx::query(
            r#"
            INSERT INTO kb_entity_mentions
                (chunk_id, entity_id, norm_start, norm_end, surface, confidence)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
        )
        .bind(chunk_id)
        .bind(entity_id)
        .bind(occ.norm_start as i32)
        .bind(occ.norm_end as i32)
        .bind(&occ.surface)
        .bind(confidence)
        .execute(pool)
        .await?;
        written += 1;
    }
    Ok(written)
}

async fn fetch_entity_labels(
    pool: &PgPool,
    ids: &[Uuid],
) -> Result<HashMap<Uuid, String>, KbError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let rows: Vec<(Uuid, String)> =
        sqlx::query_as("SELECT entity_id, label FROM kb_entities WHERE entity_id = ANY($1)")
            .bind(ids)
            .fetch_all(pool)
            .await?;
    Ok(rows.into_iter().collect())
}

fn union_aliases_with(
    canonical: &str,
    existing: &[String],
    incoming: &[String],
) -> Vec<String> {
    let mut seen_lower: HashSet<String> = HashSet::new();
    seen_lower.insert(canonical.to_lowercase());
    let mut out = Vec::new();
    for a in existing.iter().chain(incoming.iter()) {
        let key = a.to_lowercase();
        if seen_lower.insert(key) {
            out.push(a.clone());
        }
    }
    out
}

fn weighted_mean_vector(a: &[f32], na: i32, b: &[f32], nb: i32) -> Vec<f32> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() || a.len() != b.len() {
        return a.to_vec();
    }
    let total = (na + nb).max(1) as f32;
    let wa = na as f32 / total;
    let wb = nb as f32 / total;
    a.iter().zip(b.iter()).map(|(x, y)| x * wa + y * wb).collect()
}
