//! Joint entity + relation extraction for one session entry.
//!
//! This module is the LLM-touching half of the spine pipeline. The
//! entry's text → one chat call → JSON `{entities, relations}` →
//! [`RawJointExtraction`]. The persistence half (canonicalisation +
//! resolving relation labels to `entity_id`s + DB writes) lives in
//! `super::persist`.
//!
//! ### Why a separate prompt from `gw-kb::entities::extract_entities_for_chunk`
//!
//! `gw-kb` extracts entities only — that's enough for document
//! ingestion where relations between entities are inferred globally
//! from co-mention statistics by `linking.rs`. The spine wants
//! relations *as asserted in this turn* ("BM25 vs ColBERT" really
//! means `compared_with`, not just "co-mentioned"). One LLM call that
//! returns both halves keeps the entity context in working memory so
//! the relation extraction can reference the same labels — splitting
//! into two calls would either double the cost or force the second
//! call to re-extract entities.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use gw_core::{EntryType, SessionEntry};
use gw_kb::entities::{
    canonicalize_mentions, CanonicalizeOpts, RawEntity as KbRawEntity,
};
use gw_kb::ingest::KbStores;
use gw_kb::llm_parse::extract_json;
use gw_llm::{Message, OllamaClient};
use serde::Deserialize;

use super::types::{
    EntryEntityLink, EntryExtraction, EntryRelation, RawEntryEntity, RawJointExtraction,
    RawRelation, RECOMMENDED_PREDICATES,
};
use crate::error::LoopError;

/// Recommended kind taxonomy. Matches `gw_kb::entities::RECOMMENDED_KINDS`
/// so chat-extracted entities canonicalise into the same `kb_entities`
/// rows as document-ingested ones.
const RECOMMENDED_KINDS: &[&str] = &["author", "concept", "method", "dataset", "venue"];

/// Truncate the entry's text to keep a single call under typical
/// context. Chat entries are usually short, but a long agent
/// narration can spill — and entity density falls off after the first
/// ~3000 chars anyway.
const ENTRY_BUDGET_CHARS: usize = 3000;

/// Owner of the LLM client + KB stores needed to extract one entry.
/// Cheap to clone (everything inside is `Arc`-shared); the
/// `ConversationLoop` holds a single `SpineExtractor` and clones it
/// into each `tokio::spawn` task.
#[derive(Clone)]
pub struct SpineExtractor {
    pub(crate) stores: Arc<KbStores>,
}

impl SpineExtractor {
    pub fn new(stores: Arc<KbStores>) -> Self {
        Self { stores }
    }

    /// Run extraction over one entry: joint LLM call → canonicalise
    /// each entity into `kb_entities` → resolve relation endpoints
    /// to `entity_id`s → return the typed `EntryExtraction`.
    /// Returns an empty `EntryExtraction` for entry types that
    /// aren't user-facing prose (code execution, host calls, repl
    /// snapshots, compactions). The persistence step is a no-op on
    /// empty extractions, so callers can fire this for every entry
    /// without guarding.
    pub async fn extract_entry(
        &self,
        entry: &SessionEntry,
    ) -> Result<EntryExtraction, LoopError> {
        let raw = self.raw_extract_entry(entry).await?;
        self.build_typed_extraction(entry.id, raw).await
    }

    /// Test/replay seam: run the post-LLM half of `extract_entry` on
    /// a hand-crafted `RawJointExtraction`. Useful for integration
    /// tests that exercise canonicalisation + relation resolution +
    /// row construction against a real DB without paying the LLM
    /// call. Production code should always go through
    /// `extract_entry`.
    pub async fn extract_entry_from_raw(
        &self,
        entry_id: gw_core::EntryId,
        raw: RawJointExtraction,
    ) -> Result<EntryExtraction, LoopError> {
        self.build_typed_extraction(entry_id, raw).await
    }

    async fn build_typed_extraction(
        &self,
        entry_id: gw_core::EntryId,
        raw: RawJointExtraction,
    ) -> Result<EntryExtraction, LoopError> {
        if raw.entities.is_empty() {
            // Empty entity list ⇒ no relations either (parser drops
            // relations whose endpoints don't appear in the entity
            // list). Skip the canonicalisation hop.
            return Ok(EntryExtraction::default());
        }

        // Canonicalise entities through gw-kb so chat-extracted
        // entities share `kb_entities` rows with document-extracted
        // ones. Each `raw.entities[i]` resolves to `entity_ids[i]`.
        let kb_mentions: Vec<KbRawEntity> = raw
            .entities
            .iter()
            .map(|e| KbRawEntity {
                label: e.label.clone(),
                kind: e.kind.clone(),
                canonical_form: e.canonical_form.clone(),
                confidence: e.confidence,
            })
            .collect();
        let canon_opts = CanonicalizeOpts::default();
        let (entity_ids, _ingest_report) =
            canonicalize_mentions(&self.stores, &kb_mentions, &canon_opts)
                .await
                .map_err(|e| LoopError::Spine(format!("canonicalize: {e}")))?;
        debug_assert_eq!(entity_ids.len(), raw.entities.len());

        // Resolve relation endpoints. Re-build the same lookup the
        // parser used so subject/object → entity-position → entity_id.
        // The parser already dropped unresolved relations, so every
        // surviving raw.relations[*] is guaranteed to find both
        // endpoints.
        let mut entity_lookup: HashMap<String, usize> = HashMap::new();
        for (i, e) in raw.entities.iter().enumerate() {
            entity_lookup.entry(e.label.to_lowercase()).or_insert(i);
            entity_lookup
                .entry(e.canonical_form.to_lowercase())
                .or_insert(i);
        }

        let entity_ids_slice = entity_ids.as_slice();
        let entities: Vec<EntryEntityLink> = raw
            .entities
            .iter()
            .enumerate()
            .map(|(i, e)| EntryEntityLink {
                entry_id,
                entity_id: entity_ids_slice[i],
                surface: e.label.clone(),
                role: e.role.clone(),
                status: e.status.clone(),
                confidence: e.confidence,
                span_start: e.span_start,
                span_end: e.span_end,
            })
            .collect();

        let mut relations: Vec<EntryRelation> = Vec::with_capacity(raw.relations.len());
        for r in &raw.relations {
            let subj_idx = entity_lookup.get(&r.subject.to_lowercase()).copied();
            let obj_idx = entity_lookup.get(&r.object.to_lowercase()).copied();
            let (Some(si), Some(oi)) = (subj_idx, obj_idx) else {
                continue;
            };
            // Defensive: parser already drops self-loops, but
            // canonicalisation could collapse two distinct labels
            // into the same entity_id (rare but possible — same
            // surface form passed in twice with different casing
            // would canonicalise to one row).
            if entity_ids_slice[si] == entity_ids_slice[oi] {
                continue;
            }
            relations.push(EntryRelation {
                entry_id,
                subject_id: entity_ids_slice[si],
                object_id: entity_ids_slice[oi],
                predicate: r.predicate.clone(),
                directed: r.directed,
                surface: r.surface.clone(),
                confidence: r.confidence,
                span_start: r.span_start,
                span_end: r.span_end,
            });
        }

        Ok(EntryExtraction {
            entities,
            relations,
        })
    }

    /// Pre-canonicalisation extraction. Public so tests + the
    /// upcoming canonicalisation layer can call it without going
    /// through the full pipeline.
    pub async fn raw_extract_entry(
        &self,
        entry: &SessionEntry,
    ) -> Result<RawJointExtraction, LoopError> {
        let (text, role) = match &entry.entry_type {
            EntryType::UserMessage(s) => (s.as_str(), "user"),
            EntryType::AssistantMessage { content, .. } => (content.as_str(), "assistant"),
            // Skip everything else — code, host-fn results, snapshots,
            // compaction summaries, branch summaries, system messages.
            // These are either non-prose or synthesised for the agent's
            // own bookkeeping; running an extractor over them adds
            // noise to the entity graph.
            _ => return Ok(RawJointExtraction::default()),
        };
        if text.trim().is_empty() {
            return Ok(RawJointExtraction::default());
        }

        // entry id is intentionally unused at this layer — we
        // re-attach it during the typed-row build in extract_entry.
        let _ = entry.id;
        joint_extract_for_text(self.stores.llm.as_ref(), text, role).await
    }
}

/// Run the joint extractor against arbitrary text. Reusable from
/// tests with a mock `OllamaClient`. `role` is the speaker
/// (`"user"` or `"assistant"`) — the prompt uses it to bias the
/// extractor toward the right `role` field on each mention
/// ("introduced" leans user; "decided" leans assistant; both are
/// acceptable from either).
pub async fn joint_extract_for_text(
    llm: &OllamaClient,
    text: &str,
    role: &str,
) -> Result<RawJointExtraction, LoopError> {
    let prompt = build_joint_prompt(text, role);
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
        .map_err(|e| LoopError::Spine(format!("spine joint extract: {e}")))?;
    parse_joint_output(&resp.content)
        .map_err(|e| LoopError::Spine(format!("spine joint parse: {e}")))
}

const SYSTEM_PROMPT: &str = "You extract typed named entities AND the typed relations between \
them from one chat turn. Output JSON only — no prose, no markdown fences, no commentary.";

fn build_joint_prompt(text: &str, role: &str) -> String {
    let kinds = RECOMMENDED_KINDS.join(", ");
    let predicate_lines: String = RECOMMENDED_PREDICATES
        .iter()
        .map(|p| {
            format!(
                "  - {} ({}): {}\n",
                p.name,
                if p.directed { "directed" } else { "symmetric" },
                p.gloss,
            )
        })
        .collect();
    let truncated = if text.len() <= ENTRY_BUDGET_CHARS {
        text.to_string()
    } else {
        // char-boundary safe: take(N).collect() respects multibyte
        // chars (e.g. "Müller").
        let mut s: String = text.chars().take(ENTRY_BUDGET_CHARS).collect();
        s.push_str("\n…[truncated]");
        s
    };
    format!(
        "Speaker: {role}\n\
        Turn:\n\
        \"\"\"\n\
        {truncated}\n\
        \"\"\"\n\n\
        Extract two things:\n\n\
        1) NAMED entities literally mentioned in the turn. For each, choose `kind` from \
        {{{kinds}}}. Skip generic terms (\"the model\", \"our method\"), pronouns, and \
        anaphoric references. `surface` is the form as it appears; `canonical_form` is \
        the most complete / standard form (e.g. \"Patrick Lewis\" rather than \"P. Lewis\"; \
        \"Retrieval-Augmented Generation\" rather than \"RAG\"); when you can't improve on \
        the surface, repeat it. `role` is what the turn was DOING with the entity — pick \
        from {{introduced, referenced, decided, compared}} and default to \"referenced\". \
        `confidence` ∈ [0, 1]. `span_start` and `span_end` are character offsets within \
        the turn text bracketing the surface; if you cannot pin them precisely, set both \
        to null.\n\n\
        2) RELATIONS asserted between two of those entities IN THIS TURN. Only emit a \
        relation if both endpoints appear in your `entities` list. Pick `predicate` from \
        the recommended set (use a different short snake_case predicate only if none of \
        these fit):\n\
        {predicate_lines}\n\
        For symmetric predicates set `directed=false`; otherwise `subject` is the side \
        named by the predicate's gloss. `surface` is the span that asserted the relation \
        (e.g. \"BM25 vs ColBERT\" or \"ColBERT then cross-encoder\"). `confidence` ∈ [0, 1]. \
        Skip relations the turn does not actually claim — better empty than hallucinated.\n\n\
        Output JSON only, no other text:\n\
        {{\"entities\": [{{\"label\": \"...\", \"kind\": \"...\", \"canonical_form\": \"...\", \
        \"role\": \"...\", \"status\": \"mentioned\", \"confidence\": 0.0, \"span_start\": null, \
        \"span_end\": null}}], \"relations\": [{{\"subject\": \"...\", \"object\": \"...\", \
        \"predicate\": \"...\", \"directed\": true, \"surface\": \"...\", \"confidence\": 0.0, \
        \"span_start\": null, \"span_end\": null}}]}}",
    )
}

#[derive(Debug, Default, Deserialize)]
struct JointWire {
    #[serde(default)]
    entities: Vec<EntityWire>,
    #[serde(default)]
    relations: Vec<RelationWire>,
}

#[derive(Debug, Default, Deserialize)]
struct EntityWire {
    // String fields use the same null-tolerant deserialiser as
    // RelationWire — local LLMs occasionally emit nulls and we'd
    // rather drop bad rows post-parse than reject the whole payload.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    label: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    kind: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    canonical_form: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    role: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    status: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    span_start: Option<i64>,
    #[serde(default)]
    span_end: Option<i64>,
}

/// Accept a JSON `null` for a string field as if it were `""`.
/// Local LLMs occasionally emit nulls where the prompt asks for a
/// string; without this, serde rejects the entire JSON object and
/// we lose every entity in the chunk along with the bad row.
/// Downstream filters drop empty strings cleanly.
fn deserialize_optional_string<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(d)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Debug, Default, Deserialize)]
struct RelationWire {
    // Subject/object accept null because local LLMs occasionally emit
    // `"subject": null` when the prompt asks for a relation between
    // two entities the model couldn't actually identify. Without this,
    // serde rejects the entire `{entities, relations}` payload —
    // costing every entity in the same chunk too. Treat null as
    // empty; downstream filters drop the relation cleanly.
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    subject: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    object: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    predicate: String,
    #[serde(default)]
    directed: Option<bool>,
    #[serde(default)]
    surface: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    span_start: Option<i64>,
    #[serde(default)]
    span_end: Option<i64>,
}

/// Parse + validate the joint LLM output. Filters:
///   - empty/whitespace labels, empty kinds.
///   - kinds not in `RECOMMENDED_KINDS` (the prompt steers strictly,
///     so anything else is almost always a hallucinated category).
///   - within-turn duplicate `(label.to_lowercase(), kind)` entity
///     pairs — same dedup convention as `gw-kb::entities`.
///   - relations whose subject/object don't appear in the entity
///     list (case-insensitive match by label OR canonical_form).
///     Doc §4.1 step 3 — drop unresolved rather than risk wrong
///     references.
pub fn parse_joint_output(raw: &str) -> Result<RawJointExtraction, String> {
    let parsed: JointWire =
        extract_json(raw).map_err(|e| format!("joint parse: {e}"))?;

    let mut seen_entity: HashSet<(String, String)> = HashSet::new();
    let mut entities: Vec<RawEntryEntity> = Vec::with_capacity(parsed.entities.len());
    // Lowercased label OR canonical_form → index into `entities`. Lets
    // relation resolution detect "subject and object are the same
    // entity, just named two different ways" → drop as self-loop.
    let mut entity_lookup: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for e in parsed.entities {
        let label = e.label.trim().to_string();
        let kind = e.kind.trim().to_lowercase();
        if label.is_empty() || kind.is_empty() {
            continue;
        }
        if !RECOMMENDED_KINDS.contains(&kind.as_str()) {
            continue;
        }
        let dedup_key = (label.to_lowercase(), kind.clone());
        if !seen_entity.insert(dedup_key) {
            continue;
        }
        let canonical_form = if e.canonical_form.trim().is_empty() {
            label.clone()
        } else {
            e.canonical_form.trim().to_string()
        };
        let role = if e.role.trim().is_empty() {
            "referenced".to_string()
        } else {
            e.role.trim().to_lowercase()
        };
        let status = if e.status.trim().is_empty() {
            "mentioned".to_string()
        } else {
            e.status.trim().to_lowercase()
        };
        let confidence = e.confidence.map(|f| f as f32).unwrap_or(0.5).clamp(0.0, 1.0);
        // Track both label and canonical for relation resolution
        // matching, both lowercased. Position in `entities` is the
        // entity's identity for self-loop detection.
        let idx = entities.len();
        entity_lookup.entry(label.to_lowercase()).or_insert(idx);
        entity_lookup
            .entry(canonical_form.to_lowercase())
            .or_insert(idx);
        entities.push(RawEntryEntity {
            label,
            kind,
            canonical_form,
            role,
            status,
            confidence,
            span_start: e.span_start.map(|n| n as i32),
            span_end: e.span_end.map(|n| n as i32),
        });
    }

    let mut relations: Vec<RawRelation> = Vec::with_capacity(parsed.relations.len());
    for r in parsed.relations {
        let subject = r.subject.trim().to_string();
        let object = r.object.trim().to_string();
        let predicate = r.predicate.trim().to_lowercase();
        if subject.is_empty() || object.is_empty() || predicate.is_empty() {
            continue;
        }
        // Both endpoints must resolve against the entity list (by
        // label OR canonical_form). Drop unresolved relations rather
        // than risk pointing at the wrong entity in step C.
        let subj_idx = entity_lookup.get(&subject.to_lowercase()).copied();
        let obj_idx = entity_lookup.get(&object.to_lowercase()).copied();
        let (Some(si), Some(oi)) = (subj_idx, obj_idx) else {
            continue;
        };
        // Self-loops (subject and object resolve to the same entity,
        // possibly via different surface/canonical forms) carry no
        // signal — drop.
        if si == oi {
            continue;
        }
        let directed = r.directed.unwrap_or(true);
        let confidence = r.confidence.map(|f| f as f32).unwrap_or(0.5).clamp(0.0, 1.0);
        let surface = if r.surface.trim().is_empty() {
            // Fall back to a synthesised surface so the schema's NOT
            // NULL constraint on `surface` is satisfied even when the
            // model omitted the span.
            format!("{subject} {predicate} {object}")
        } else {
            r.surface.trim().to_string()
        };
        relations.push(RawRelation {
            subject,
            object,
            predicate,
            directed,
            surface,
            confidence,
            span_start: r.span_start.map(|n| n as i32),
            span_end: r.span_end.map(|n| n as i32),
        });
    }

    Ok(RawJointExtraction { entities, relations })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(label: &str, kind: &str) -> RawEntryEntity {
        RawEntryEntity {
            label: label.into(),
            kind: kind.into(),
            canonical_form: label.into(),
            role: "referenced".into(),
            status: "mentioned".into(),
            confidence: 0.9,
            span_start: None,
            span_end: None,
        }
    }

    #[test]
    fn happy_path_parses_entities_and_relations() {
        let raw = r#"{
            "entities": [
                {"label": "BM25", "kind": "method", "canonical_form": "BM25",
                 "role": "compared", "status": "mentioned", "confidence": 0.95},
                {"label": "ColBERT", "kind": "method", "canonical_form": "ColBERT",
                 "role": "compared", "status": "mentioned", "confidence": 0.95}
            ],
            "relations": [
                {"subject": "BM25", "object": "ColBERT", "predicate": "compared_with",
                 "directed": false, "surface": "BM25 vs ColBERT", "confidence": 0.9}
            ]
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities.len(), 2);
        assert_eq!(out.relations.len(), 1);
        assert_eq!(out.relations[0].predicate, "compared_with");
        assert!(!out.relations[0].directed);
    }

    #[test]
    fn drops_relations_with_unresolved_endpoints() {
        // "FAISS" never appears in the entity list — relation must be dropped.
        let raw = r#"{
            "entities": [
                {"label": "ColBERT", "kind": "method", "canonical_form": "ColBERT",
                 "role": "referenced", "status": "mentioned", "confidence": 0.9}
            ],
            "relations": [
                {"subject": "ColBERT", "object": "FAISS", "predicate": "uses",
                 "directed": true, "surface": "ColBERT uses FAISS", "confidence": 0.7}
            ]
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities.len(), 1);
        assert!(out.relations.is_empty());
    }

    #[test]
    fn relation_resolves_via_canonical_form() {
        // Subject is "RAG" (matches surface label); object is
        // "Retrieval-Augmented Generation" (matches canonical_form).
        let raw = r#"{
            "entities": [
                {"label": "RAG", "kind": "method",
                 "canonical_form": "Retrieval-Augmented Generation",
                 "role": "referenced", "status": "mentioned", "confidence": 0.95}
            ],
            "relations": [
                {"subject": "RAG", "object": "Retrieval-Augmented Generation",
                 "predicate": "is_a", "directed": true,
                 "surface": "RAG is Retrieval-Augmented Generation", "confidence": 0.9}
            ]
        }"#;
        let out = parse_joint_output(raw).unwrap();
        // Self-loop: subject and object resolve to the same entity.
        // Dropped to avoid noise.
        assert!(out.relations.is_empty());
    }

    #[test]
    fn drops_invalid_kinds() {
        let raw = r#"{
            "entities": [
                {"label": "ColBERT", "kind": "method", "canonical_form": "ColBERT",
                 "role": "referenced", "status": "mentioned", "confidence": 0.9},
                {"label": "FAISS", "kind": "tool", "canonical_form": "FAISS",
                 "role": "referenced", "status": "mentioned", "confidence": 0.9}
            ],
            "relations": []
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].label, "ColBERT");
    }

    #[test]
    fn dedups_within_turn_case_insensitive() {
        let raw = r#"{
            "entities": [
                {"label": "BERT", "kind": "method", "canonical_form": "BERT",
                 "role": "referenced", "status": "mentioned", "confidence": 0.9},
                {"label": "bert", "kind": "method", "canonical_form": "BERT",
                 "role": "referenced", "status": "mentioned", "confidence": 0.6}
            ],
            "relations": []
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities.len(), 1);
    }

    #[test]
    fn tolerates_null_string_fields_from_local_llm() {
        // Real failure mode from qwen3.5:9b — relation endpoints come
        // back as null when the LLM couldn't identify them. Before
        // the fix, serde rejected the whole payload and we lost the
        // valid entity too. After: bad relation drops, entity stays.
        let raw = r#"{
            "entities": [{"label": "Survey on RAG", "kind": "venue",
                          "canonical_form": "Survey on RAG", "role": "referenced",
                          "status": "mentioned", "confidence": 0.9}],
            "relations": [{"subject": null, "object": null,
                           "predicate": "compared_with", "directed": false,
                           "surface": "compared with neighbours", "confidence": 0.8}]
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities.len(), 1);
        assert_eq!(out.entities[0].label, "Survey on RAG");
        // Relation has empty subject/object after null → fails the
        // empty-string filter and gets dropped. No relations land.
        assert!(out.relations.is_empty());
    }

    #[test]
    fn empty_payload_is_ok() {
        let raw = r#"{"entities": [], "relations": []}"#;
        let out = parse_joint_output(raw).unwrap();
        assert!(out.entities.is_empty());
        assert!(out.relations.is_empty());
    }

    #[test]
    fn tolerates_markdown_fences_and_preamble() {
        let raw = "Sure, here's the JSON:\n```json\n{\"entities\": [{\"label\": \"BERT\", \
            \"kind\": \"method\", \"canonical_form\": \"BERT\", \"role\": \"referenced\", \
            \"status\": \"mentioned\", \"confidence\": 0.8}], \"relations\": []}\n```";
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities[0].label, "BERT");
    }

    #[test]
    fn synthesises_surface_when_missing() {
        let raw = r#"{
            "entities": [
                {"label": "BM25", "kind": "method", "canonical_form": "BM25",
                 "role": "compared", "status": "mentioned", "confidence": 0.9},
                {"label": "ColBERT", "kind": "method", "canonical_form": "ColBERT",
                 "role": "compared", "status": "mentioned", "confidence": 0.9}
            ],
            "relations": [
                {"subject": "BM25", "object": "ColBERT", "predicate": "compared_with",
                 "directed": false, "surface": "", "confidence": 0.7}
            ]
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.relations.len(), 1);
        assert!(out.relations[0].surface.contains("BM25"));
        assert!(out.relations[0].surface.contains("ColBERT"));
    }

    #[test]
    fn defaults_missing_role_and_status_and_confidence() {
        let raw = r#"{
            "entities": [
                {"label": "BERT", "kind": "method", "canonical_form": "BERT"}
            ],
            "relations": []
        }"#;
        let out = parse_joint_output(raw).unwrap();
        assert_eq!(out.entities[0].role, "referenced");
        assert_eq!(out.entities[0].status, "mentioned");
        assert!((out.entities[0].confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn entity_struct_helper_is_used() {
        // Ensure the test helper compiles with the public type so
        // future tests can build raw extractions inline.
        let e = ent("BM25", "method");
        assert_eq!(e.kind, "method");
    }
}
