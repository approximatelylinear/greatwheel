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

use gw_llm::{Message, OllamaClient};
use serde::{Deserialize, Serialize};

use crate::error::KbError;
use crate::llm_parse::extract_json;

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
    parse_extractor_output(&resp.content)
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
        - Skip generic terms (\"the model\", \"our method\") that aren't names.\n\
        - Skip pronouns and anaphoric references.\n\n\
        Passage:\n\
        \"\"\"\n\
        {content_truncated}\n\
        \"\"\"\n\n\
        Output JSON only, no other text:\n\
        {{\"entities\": [{{\"label\": \"...\", \"kind\": \"author|concept|method|dataset|venue\", \
        \"canonical_form\": \"...\", \"confidence\": 0.0}}]}}",
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
        out.push(RawEntity {
            label,
            kind,
            canonical_form,
            confidence,
        });
    }
    Ok(out)
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
}
