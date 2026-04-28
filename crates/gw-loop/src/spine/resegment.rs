//! DB-backed orchestrator for spine segmentation.
//!
//! Wraps the pure algorithm in `super::segment` with the I/O it
//! needs: load entries + their entity links from Postgres, run
//! `segment()`, label new ranges via the LLM, diff against the
//! cached `session_segments` rows, and persist.
//!
//! Diff strategy (matches design-semantic-spine.md §4.2 step 5):
//! a previously-cached segment whose `(entry_first, entry_last)`
//! range matches a new proposal is *kept* (its `segment_id` survives,
//! `entity_ids` updated in place if drifted). Existing segments
//! whose ranges no longer match anything in the proposal are
//! marked `invalidated_at = now()` instead of deleted, so history
//! replay can show the labels that were live at any past point.
//! Brand-new ranges get a fresh `segment_id` and one LLM call for
//! `(label, kind)`, with a graceful fallback if the call fails.
//!
//! ### LLM cost
//!
//! One call per *new* segment, never per turn. A typical 30-turn
//! conversation produces ~5 segments — five calls amortised across
//! the whole session. The diff path skips the call when a range
//! survives, so steady-state edits during a session don't
//! re-prompt for unchanged segments.

use std::collections::{HashMap, HashSet};

use gw_core::EntryId;
use gw_kb::llm_parse::extract_json;
use gw_llm::{Message, OllamaClient};
use serde::Deserialize;
use sqlx::PgPool;
use tracing::{debug, warn};
use uuid::Uuid;

use super::segment::{segment, ProposedSegment, SegmentEntry, SegmentOpts};
use crate::error::LoopError;

/// Recommended segment kinds — what the LLM is asked to pick from.
/// Free-form `TEXT` in the schema; the prompt steers strictly here
/// and we filter unknown kinds back to `"other"` post-parse.
pub const RECOMMENDED_KINDS: &[&str] =
    &["comparison", "decision", "deep_dive", "construction", "other"];

/// Maximum entity labels included in the LLM labelling prompt. The
/// segmenter's top-N ranking by mention count keeps the most
/// representative labels first; entries past N rarely change the
/// segment's kind.
const PROMPT_ENTITY_BUDGET: usize = 8;

/// Maximum chars of first/last entry text to include in the
/// labelling prompt. Two windows × this size.
const PROMPT_TEXT_BUDGET_CHARS: usize = 600;

#[derive(Debug, Clone, Copy, Default)]
pub struct ResegmentOpts {
    pub segment: SegmentOpts,
}

#[derive(Debug, Clone, Default)]
pub struct ResegmentReport {
    pub entries_seen: usize,
    pub segments_proposed: usize,
    pub segments_created: usize,
    pub segments_updated: usize,
    pub segments_unchanged: usize,
    pub segments_invalidated: usize,
    pub llm_failures: usize,
}

/// Snapshot of one persisted segment, suitable for outbound events
/// (`LoopEvent::SegmentsUpdated` in step C consumes this).
#[derive(Debug, Clone)]
pub struct SegmentSnapshot {
    pub segment_id: Uuid,
    pub session_id: Uuid,
    pub label: String,
    pub kind: String,
    pub entry_first: EntryId,
    pub entry_last: EntryId,
    pub entity_ids: Vec<Uuid>,
    pub summary: Option<String>,
}

/// Top-level entry point. Loads entries + entity links for
/// `session_id`, segments them, labels new ranges, diffs against
/// the cache, and persists. Returns a report and the current set
/// of (live) segments after persistence.
pub async fn resegment(
    pool: &PgPool,
    llm: &OllamaClient,
    session_id: Uuid,
    opts: &ResegmentOpts,
) -> Result<(ResegmentReport, Vec<SegmentSnapshot>), LoopError> {
    let mut report = ResegmentReport::default();

    let entries = load_entries_with_entity_ids(pool, session_id).await?;
    report.entries_seen = entries.len();
    if entries.is_empty() {
        // Wipe any leftovers — a session that lost all its entries
        // shouldn't keep stale segments.
        let invalidated = invalidate_unmatched(pool, session_id, &HashSet::new()).await?;
        report.segments_invalidated = invalidated;
        return Ok((report, Vec::new()));
    }

    let proposed = segment(&entries, &opts.segment);
    report.segments_proposed = proposed.len();

    let existing = load_current_segments(pool, session_id).await?;
    let existing_by_range: HashMap<(EntryId, EntryId), &CurrentSegmentRow> = existing
        .iter()
        .map(|s| ((s.entry_first, s.entry_last), s))
        .collect();

    let mut kept: HashSet<Uuid> = HashSet::new();
    let mut snapshots: Vec<SegmentSnapshot> = Vec::with_capacity(proposed.len());

    for prop in &proposed {
        let key = (prop.entry_first, prop.entry_last);
        if let Some(prev) = existing_by_range.get(&key) {
            kept.insert(prev.segment_id);
            if prev.entity_ids != prop.entity_ids {
                update_segment_entities(pool, prev.segment_id, &prop.entity_ids).await?;
                report.segments_updated += 1;
            } else {
                report.segments_unchanged += 1;
            }
            snapshots.push(SegmentSnapshot {
                segment_id: prev.segment_id,
                session_id,
                label: prev.label.clone(),
                kind: prev.kind.clone(),
                entry_first: prev.entry_first,
                entry_last: prev.entry_last,
                entity_ids: prop.entity_ids.clone(),
                summary: prev.summary.clone(),
            });
        } else {
            // Brand-new range — needs an LLM-generated label.
            let labelled = match label_segment(pool, llm, prop).await {
                Ok(l) => l,
                Err(e) => {
                    warn!(error = %e, "segment labelling failed; using fallback");
                    report.llm_failures += 1;
                    fallback_label(prop)
                }
            };
            let segment_id =
                insert_segment(pool, session_id, prop, &labelled).await?;
            kept.insert(segment_id);
            report.segments_created += 1;
            snapshots.push(SegmentSnapshot {
                segment_id,
                session_id,
                label: labelled.label,
                kind: labelled.kind,
                entry_first: prop.entry_first,
                entry_last: prop.entry_last,
                entity_ids: prop.entity_ids.clone(),
                summary: None,
            });
        }
    }

    let invalidated = invalidate_unmatched(pool, session_id, &kept).await?;
    report.segments_invalidated = invalidated;

    debug!(
        session_id = %session_id,
        entries = report.entries_seen,
        proposed = report.segments_proposed,
        created = report.segments_created,
        updated = report.segments_updated,
        invalidated = report.segments_invalidated,
        "resegment complete"
    );
    Ok((report, snapshots))
}

// ─── Loading: entries with their canonical entity_ids ───────────────

async fn load_entries_with_entity_ids(
    pool: &PgPool,
    session_id: Uuid,
) -> Result<Vec<SegmentEntry>, LoopError> {
    // One query — left join so entries with no extracted entities
    // still appear (they'll start fresh segments by themselves).
    type Row = (Uuid, Option<Uuid>);
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT se.id, ee.entity_id
        FROM session_entries se
        LEFT JOIN session_entry_entities ee ON ee.entry_id = se.id
        WHERE se.session_id = $1
          AND se.entry_type IN ('user_message', 'assistant_message')
        ORDER BY se.created_at, se.id, ee.entity_id
        "#,
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("load entries: {e}")))?;

    // Group by entry id, preserving the SQL ordering.
    let mut out: Vec<SegmentEntry> = Vec::new();
    let mut current: Option<SegmentEntry> = None;
    for (entry_id, entity_id) in rows {
        let eid = EntryId(entry_id);
        match &mut current {
            Some(s) if s.entry_id == eid => {
                if let Some(ent) = entity_id {
                    s.entity_ids.push(ent);
                }
            }
            _ => {
                if let Some(s) = current.take() {
                    out.push(s);
                }
                let mut entity_ids = Vec::new();
                if let Some(ent) = entity_id {
                    entity_ids.push(ent);
                }
                current = Some(SegmentEntry {
                    entry_id: eid,
                    entity_ids,
                });
            }
        }
    }
    if let Some(s) = current {
        out.push(s);
    }
    Ok(out)
}

// ─── Loading: current cache ────────────────────────────────────────

#[derive(Debug, Clone)]
struct CurrentSegmentRow {
    segment_id: Uuid,
    label: String,
    kind: String,
    entry_first: EntryId,
    entry_last: EntryId,
    entity_ids: Vec<Uuid>,
    summary: Option<String>,
}

async fn load_current_segments(
    pool: &PgPool,
    session_id: Uuid,
) -> Result<Vec<CurrentSegmentRow>, LoopError> {
    type Row = (Uuid, String, String, Uuid, Uuid, Vec<Uuid>, Option<String>);
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT segment_id, label, kind, entry_first, entry_last, entity_ids, summary
        FROM session_segments
        WHERE session_id = $1 AND invalidated_at IS NULL
        ORDER BY entry_first
        "#,
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("load current segments: {e}")))?;
    Ok(rows
        .into_iter()
        .map(|(segment_id, label, kind, ef, el, eids, summary)| CurrentSegmentRow {
            segment_id,
            label,
            kind,
            entry_first: EntryId(ef),
            entry_last: EntryId(el),
            entity_ids: eids,
            summary,
        })
        .collect())
}

// ─── Persistence ───────────────────────────────────────────────────

async fn insert_segment(
    pool: &PgPool,
    session_id: Uuid,
    prop: &ProposedSegment,
    labelled: &SegmentLabel,
) -> Result<Uuid, LoopError> {
    let segment_id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO session_segments
            (session_id, label, kind, entry_first, entry_last, entity_ids)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING segment_id
        "#,
    )
    .bind(session_id)
    .bind(&labelled.label)
    .bind(&labelled.kind)
    .bind(prop.entry_first.0)
    .bind(prop.entry_last.0)
    .bind(&prop.entity_ids)
    .fetch_one(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("insert segment: {e}")))?;
    Ok(segment_id)
}

async fn update_segment_entities(
    pool: &PgPool,
    segment_id: Uuid,
    entity_ids: &[Uuid],
) -> Result<(), LoopError> {
    sqlx::query("UPDATE session_segments SET entity_ids = $2 WHERE segment_id = $1")
        .bind(segment_id)
        .bind(entity_ids)
        .execute(pool)
        .await
        .map_err(|e| LoopError::Spine(format!("update segment entities: {e}")))?;
    Ok(())
}

/// Mark every current segment of `session_id` not in `kept_ids` as
/// invalidated. Returns the count invalidated.
async fn invalidate_unmatched(
    pool: &PgPool,
    session_id: Uuid,
    kept_ids: &HashSet<Uuid>,
) -> Result<usize, LoopError> {
    let kept: Vec<Uuid> = kept_ids.iter().copied().collect();
    let r = sqlx::query(
        r#"
        UPDATE session_segments
        SET invalidated_at = now()
        WHERE session_id = $1
          AND invalidated_at IS NULL
          AND segment_id <> ALL($2::uuid[])
        "#,
    )
    .bind(session_id)
    .bind(&kept)
    .execute(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("invalidate segments: {e}")))?;
    Ok(r.rows_affected() as usize)
}

// ─── LLM labelling ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SegmentLabel {
    label: String,
    kind: String,
}

fn fallback_label(prop: &ProposedSegment) -> SegmentLabel {
    let n = prop.entity_ids.len();
    SegmentLabel {
        label: if n == 0 {
            "untagged".into()
        } else if n == 1 {
            "1 entity".into()
        } else {
            format!("{n} entities")
        },
        kind: "other".into(),
    }
}

async fn label_segment(
    pool: &PgPool,
    llm: &OllamaClient,
    prop: &ProposedSegment,
) -> Result<SegmentLabel, LoopError> {
    let entity_labels =
        fetch_top_entity_labels(pool, &prop.entity_ids, PROMPT_ENTITY_BUDGET).await?;
    let first_text = fetch_entry_text(pool, prop.entry_first).await?;
    let last_text = if prop.entry_first == prop.entry_last {
        String::new() // single-entry segment — first text covers it
    } else {
        fetch_entry_text(pool, prop.entry_last).await?
    };

    let prompt = build_label_prompt(&entity_labels, &first_text, &last_text);
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
        .map_err(|e| LoopError::Spine(format!("label_segment chat: {e}")))?;
    parse_label_output(&resp.content)
}

const SYSTEM_PROMPT: &str = "You assign a short label and a kind to one segment of a chat \
conversation. Output JSON only — no prose, no markdown fences, no commentary.";

fn build_label_prompt(entity_labels: &[String], first_text: &str, last_text: &str) -> String {
    let kinds = RECOMMENDED_KINDS.join(", ");
    let entities = if entity_labels.is_empty() {
        "(none)".to_string()
    } else {
        entity_labels.join(", ")
    };
    let truncated_first = truncate(first_text, PROMPT_TEXT_BUDGET_CHARS);
    let last_block = if last_text.is_empty() {
        String::new()
    } else {
        format!(
            "\nLast entry:\n\"\"\"\n{}\n\"\"\"\n",
            truncate(last_text, PROMPT_TEXT_BUDGET_CHARS)
        )
    };
    format!(
        "Top entities in segment: {entities}\n\
        \n\
        First entry:\n\
        \"\"\"\n\
        {truncated_first}\n\
        \"\"\"\n\
        {last_block}\n\
        Pick a 1-3 word `label` (lowercase, the kind of phrase that'd appear on a research-\
        notebook tab — e.g. \"recall vs precision\", \"colbert pipeline\", \"benchmark choice\") \
        and a `kind` from {{{kinds}}}. \
        - comparison: weighing two or more options against each other. \
        - decision: picking one option over others. \
        - deep_dive: exploring one topic in depth. \
        - construction: building or composing a system. \
        - other: anything else (use sparingly).\n\
        \n\
        Output JSON only:\n\
        {{\"label\": \"...\", \"kind\": \"...\"}}",
    )
}

/// Best-effort repair for a known qwen3.5:9b quirk: it sometimes
/// drops the *opening* quote on JSON keys, emitting things like
/// `{label": "cluster pipeline", kind": "construction"}`. The
/// closing quote and colon are still there. We walk the string and,
/// at every position immediately after `{` or `,` (skipping
/// whitespace), if we see an unquoted identifier followed by `":`,
/// we splice in the missing opening quote. Strings already inside
/// quotes are left untouched.
///
/// Pure prefix-of-key heuristic — won't repair structural breakage
/// (missing braces, unbalanced strings). The caller treats the
/// result as a "second-chance parse"; if it still fails, the
/// labeller's fallback path kicks in.
fn repair_unquoted_keys(raw: &str) -> String {
    // The structural characters we look at (`{`, `,`, `"`, `:`,
    // identifier chars, whitespace) are all ASCII-single-byte, so
    // we can scan with byte positions but must rebuild the output
    // using `&str` slicing — that keeps multibyte chars in label
    // values intact.
    let b = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() + 16);
    let mut i = 0;
    let mut in_string = false;
    let mut last_flush = 0;
    while i < b.len() {
        if in_string {
            // Skip past the closing quote, honoring `\"` escapes.
            if b[i] == b'\\' && i + 1 < b.len() {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b[i] == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if b[i] == b'{' || b[i] == b',' {
            // Look ahead past whitespace for an unquoted identifier
            // followed by `":` (qwen3.5's missing-opening-quote quirk).
            let mut j = i + 1;
            while j < b.len() && (b[j] as char).is_ascii_whitespace() {
                j += 1;
            }
            let id_start = j;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > id_start && j + 1 < b.len() && b[j] == b'"' && b[j + 1] == b':' {
                // Splice in the missing opening quote at id_start.
                // The `"` we found at j is now the *closing* quote of
                // the key — flip in_string=true so the main loop's
                // handling of b[j] (which it'll encounter next iter
                // via i = j) correctly treats it as a closer.
                out.push_str(&raw[last_flush..id_start]);
                out.push('"');
                last_flush = id_start;
                in_string = true;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out.push_str(&raw[last_flush..]);
    out
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("\n…[truncated]");
    out
}

#[derive(Debug, Default, Deserialize)]
struct LabelWire {
    #[serde(default)]
    label: String,
    #[serde(default)]
    kind: String,
}

fn parse_label_output(raw: &str) -> Result<SegmentLabel, LoopError> {
    let parsed: LabelWire = extract_json::<LabelWire>(raw)
        .or_else(|_| extract_json::<LabelWire>(&repair_unquoted_keys(raw)))
        .map_err(|e| LoopError::Spine(format!("parse label: {e}")))?;
    let label = parsed.label.trim().to_string();
    if label.is_empty() {
        return Err(LoopError::Spine(
            "label_segment returned empty label".into(),
        ));
    }
    let kind_raw = parsed.kind.trim().to_lowercase();
    let kind = if RECOMMENDED_KINDS.contains(&kind_raw.as_str()) {
        kind_raw
    } else {
        // Unknown kind — keep a useful fallback rather than reject.
        // doc §3.3 lists the recommended set; out-of-vocab values
        // collapse to "other" so the kind column stays a useful filter.
        "other".to_string()
    };
    Ok(SegmentLabel { label, kind })
}

// ─── Lookups for the labeller ──────────────────────────────────────

async fn fetch_entry_text(pool: &PgPool, entry_id: EntryId) -> Result<String, LoopError> {
    let row: Option<(serde_json::Value,)> =
        sqlx::query_as("SELECT content FROM session_entries WHERE id = $1")
            .bind(entry_id.0)
            .fetch_optional(pool)
            .await
            .map_err(|e| LoopError::Spine(format!("fetch entry text: {e}")))?;
    let Some((content,)) = row else {
        return Ok(String::new());
    };
    // EntryType::UserMessage(s) and AssistantMessage { content, .. }
    // serialise as { "UserMessage": "..." } and
    // { "AssistantMessage": { "content": "...", ... } } respectively.
    if let Some(s) = content.get("UserMessage").and_then(|v| v.as_str()) {
        return Ok(s.to_string());
    }
    if let Some(s) = content
        .get("AssistantMessage")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        return Ok(s.to_string());
    }
    Ok(String::new())
}

async fn fetch_top_entity_labels(
    pool: &PgPool,
    entity_ids: &[Uuid],
    n: usize,
) -> Result<Vec<String>, LoopError> {
    let take = entity_ids.len().min(n);
    if take == 0 {
        return Ok(Vec::new());
    }
    let slice = &entity_ids[..take];
    // Query returns a flat list; we re-order to match `slice` so the
    // labeller's ranking-by-mention-count input order is preserved.
    let rows: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT entity_id, label FROM kb_entities WHERE entity_id = ANY($1)",
    )
    .bind(slice)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch entity labels: {e}")))?;
    let by_id: HashMap<Uuid, String> = rows.into_iter().collect();
    Ok(slice
        .iter()
        .filter_map(|id| by_id.get(id).cloned())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_label_uses_entity_count() {
        let prop = ProposedSegment {
            entry_first: EntryId(Uuid::nil()),
            entry_last: EntryId(Uuid::nil()),
            entity_ids: vec![Uuid::new_v4(); 5],
            entry_ids: vec![EntryId(Uuid::nil())],
        };
        let l = fallback_label(&prop);
        assert_eq!(l.label, "5 entities");
        assert_eq!(l.kind, "other");
    }

    #[test]
    fn parse_label_filters_unknown_kind_to_other() {
        let raw = r#"{"label": "recall vs precision", "kind": "tradeoff"}"#;
        let l = parse_label_output(raw).unwrap();
        assert_eq!(l.label, "recall vs precision");
        assert_eq!(l.kind, "other");
    }

    #[test]
    fn parse_label_keeps_known_kind() {
        let raw = r#"{"label": "bm25 vs colbert", "kind": "comparison"}"#;
        let l = parse_label_output(raw).unwrap();
        assert_eq!(l.kind, "comparison");
    }

    #[test]
    fn parse_label_rejects_empty_label() {
        let raw = r#"{"label": "", "kind": "comparison"}"#;
        assert!(parse_label_output(raw).is_err());
    }

    #[test]
    fn parse_label_tolerates_fences() {
        let raw = "```json\n{\"label\": \"colbert pipeline\", \"kind\": \"construction\"}\n```";
        let l = parse_label_output(raw).unwrap();
        assert_eq!(l.label, "colbert pipeline");
    }

    #[test]
    fn repairs_qwen_missing_opening_key_quotes() {
        // The exact failure mode observed in production: qwen3.5:9b
        // dropping the opening quote on each key. extract_json fails
        // on the raw form; the repair pass restores `"key":` so the
        // second-chance parse in parse_label_output succeeds.
        let raw = r#"{label": "cluster pipeline", kind": "construction"}"#;
        let repaired = repair_unquoted_keys(raw);
        assert_eq!(
            repaired,
            r#"{"label": "cluster pipeline", "kind": "construction"}"#
        );
        let parsed = parse_label_output(raw).unwrap();
        assert_eq!(parsed.label, "cluster pipeline");
        assert_eq!(parsed.kind, "construction");
    }

    #[test]
    fn repair_preserves_multibyte_chars_in_values() {
        // Sanity: multibyte values (labels containing accented
        // names, em-dashes, etc.) survive the byte-level scan.
        let raw = r#"{label": "Müller — recall", kind": "comparison"}"#;
        let repaired = repair_unquoted_keys(raw);
        assert!(repaired.contains("Müller — recall"));
        assert!(repaired.contains(r#""label":"#));
        assert!(repaired.contains(r#""kind":"#));
    }

    #[test]
    fn repair_is_noop_on_well_formed_json() {
        let raw = r#"{"label": "ok", "kind": "decision"}"#;
        assert_eq!(repair_unquoted_keys(raw), raw);
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "Müller and Bär discuss recall vs precision";
        let t = truncate(s, 10);
        assert!(t.starts_with("Müller and"));
        assert!(t.ends_with("[truncated]"));
    }
}
