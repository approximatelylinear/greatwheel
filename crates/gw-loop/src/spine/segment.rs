//! Segmentation: one segment per turn.
//!
//! The pure half of design-semantic-spine.md §4.2 — given a list of
//! entries flagged with their turn boundary (`is_turn_start`), group
//! all entries from one turn-start (inclusive) up to the next into
//! a single segment, with the union of their entity_ids ranked by
//! mention count.
//!
//! "Turn" here = "what the user kicked off with one message." Each
//! `UserMessage` entry opens a new segment; the assistant_message,
//! assistant_narration, and code_execution entries that follow it
//! join that segment until the next user message arrives.
//!
//! This is a deliberate simplification of the earlier shared-entity
//! contiguity heuristic. In practice the semantic spine reads more
//! cleanly when each turn is one rail marker carrying everything the
//! agent did in response to one user input — pin chains, narration
//! prose, and entities all collect under a single label.
//!
//! No DB, no LLM, no embedder dependency. The orchestrator (step B)
//! loads entries + entity links from Postgres, derives `is_turn_start`
//! from `entry_type`, calls this, then labels new segments via LLM
//! and diffs against the cache.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

use gw_core::EntryId;

/// Default cap on segment length, so one ridiculously long turn
/// doesn't monopolise the whole spine. Turns rarely exceed this in
/// practice but the cap keeps the rail predictable.
pub const DEFAULT_MAX_ENTRIES_PER_SEGMENT: usize = 16;

/// Tunable knobs for the segmenter.
#[derive(Debug, Clone, Copy)]
pub struct SegmentOpts {
    pub max_entries_per_segment: usize,
}

impl Default for SegmentOpts {
    fn default() -> Self {
        Self {
            max_entries_per_segment: DEFAULT_MAX_ENTRIES_PER_SEGMENT,
        }
    }
}

/// One entry's input to segmentation: its id, the canonical
/// entity_ids appearing in it (already canonicalised through
/// `kb_entities` by the spine extractor), and whether it opens a
/// new turn (true for `UserMessage` entries, false otherwise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentEntry {
    pub entry_id: EntryId,
    pub entity_ids: Vec<Uuid>,
    /// True if this entry opens a new turn — used to mark segment
    /// boundaries. Set by the orchestrator from the entry's type
    /// (`user_message` → true, everything else → false).
    pub is_turn_start: bool,
}

/// One computed segment: a contiguous range of entries plus the
/// union of entities appearing across them, ranked by mention count.
/// `kind` is left to the LLM-labelling step (B); the pure algorithm
/// has no opinion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedSegment {
    pub entry_first: EntryId,
    pub entry_last: EntryId,
    /// All entities mentioned across the segment, deduplicated and
    /// sorted by descending mention count (ties broken by insertion
    /// order). The labeller (step B) takes the top N when prompting.
    pub entity_ids: Vec<Uuid>,
    /// Entries-in-order. Useful for the labeller's "first/last entry
    /// text" prompt input without re-querying.
    pub entry_ids: Vec<EntryId>,
}

/// Segment a sequence of entries by turn boundary. Returns
/// segments in entry order. Empty input → empty output.
///
/// Algorithm (single-pass):
///   1. The first entry seeds segment 0.
///   2. For each subsequent entry: if `is_turn_start` is true OR the
///      current segment has hit `max_entries_per_segment`, close the
///      current segment and start a new one with this entry.
///      Otherwise append.
///
/// Entries before the first turn-start (rare — typically a system
/// banner) collect into a leading segment that closes as soon as the
/// first user message arrives.
pub fn segment(entries: &[SegmentEntry], opts: &SegmentOpts) -> Vec<ProposedSegment> {
    if entries.is_empty() {
        return Vec::new();
    }

    let mut segments: Vec<ProposedSegment> = Vec::new();
    let mut cur_entry_ids: Vec<EntryId> = Vec::new();
    let mut cur_set: HashSet<Uuid> = HashSet::new();
    // Mention counts per entity within the current segment, used to
    // rank the entity_ids field at segment close.
    let mut cur_counts: std::collections::HashMap<Uuid, (u32, u32)> =
        std::collections::HashMap::new();
    let mut seq: u32 = 0;

    for e in entries {
        let must_break = !cur_entry_ids.is_empty()
            && (e.is_turn_start
                || cur_entry_ids.len() >= opts.max_entries_per_segment);

        if must_break {
            segments.push(close_segment(&cur_entry_ids, &cur_counts));
            cur_entry_ids.clear();
            cur_set.clear();
            cur_counts.clear();
        }

        cur_entry_ids.push(e.entry_id);
        for eid in &e.entity_ids {
            cur_set.insert(*eid);
            cur_counts
                .entry(*eid)
                .and_modify(|c| c.0 += 1)
                .or_insert_with(|| {
                    seq += 1;
                    (1, seq)
                });
        }
    }

    if !cur_entry_ids.is_empty() {
        segments.push(close_segment(&cur_entry_ids, &cur_counts));
    }

    segments
}

fn close_segment(
    entry_ids: &[EntryId],
    counts: &std::collections::HashMap<Uuid, (u32, u32)>,
) -> ProposedSegment {
    let mut ranked: Vec<(Uuid, u32, u32)> =
        counts.iter().map(|(k, (c, s))| (*k, *c, *s)).collect();
    // count desc, then first-seen asc for stability
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    let entity_ids: Vec<Uuid> = ranked.into_iter().map(|(eid, _, _)| eid).collect();
    ProposedSegment {
        entry_first: entry_ids[0],
        entry_last: *entry_ids.last().expect("non-empty"),
        entity_ids,
        entry_ids: entry_ids.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(n: usize) -> Uuid {
        // Deterministic, distinct UUIDs for tests.
        let mut bytes = [0u8; 16];
        bytes[15] = n as u8;
        Uuid::from_bytes(bytes)
    }

    fn entry(id: u8, ents: &[usize], is_turn_start: bool) -> SegmentEntry {
        let mut bytes = [0u8; 16];
        bytes[0] = id;
        SegmentEntry {
            entry_id: EntryId(Uuid::from_bytes(bytes)),
            entity_ids: ents.iter().map(|n| ent(*n)).collect(),
            is_turn_start,
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let out = segment(&[], &SegmentOpts::default());
        assert!(out.is_empty());
    }

    #[test]
    fn single_entry_makes_singleton_segment() {
        let entries = vec![entry(1, &[1, 2, 3], true)];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_ids.len(), 1);
        assert_eq!(out[0].entity_ids.len(), 3);
    }

    #[test]
    fn one_turn_collects_all_followers() {
        // user → assistant → narration → code: all one segment.
        let entries = vec![
            entry(1, &[1], true),     // UserMessage
            entry(2, &[1, 2], false), // AssistantMessage
            entry(3, &[2, 3], false), // AssistantNarration
            entry(4, &[3], false),    // CodeExecution (entity-less in practice)
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_ids.len(), 4);
        // Union: {1,2,3}; counts {1:2, 2:2, 3:2}; tiebreak first-seen.
        assert_eq!(out[0].entity_ids, vec![ent(1), ent(2), ent(3)]);
    }

    #[test]
    fn second_user_message_opens_new_segment() {
        let entries = vec![
            entry(1, &[1, 2], true),  // turn 1 user
            entry(2, &[1, 3], false), // turn 1 assistant
            entry(3, &[4, 5], true),  // turn 2 user
            entry(4, &[4, 6], false), // turn 2 assistant
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].entry_ids.len(), 2);
        assert_eq!(out[1].entry_ids.len(), 2);
        // Each segment carries only its own turn's entities.
        assert!(out[0].entity_ids.contains(&ent(1)));
        assert!(out[0].entity_ids.contains(&ent(3)));
        assert!(!out[0].entity_ids.contains(&ent(4)));
        assert!(out[1].entity_ids.contains(&ent(4)));
        assert!(out[1].entity_ids.contains(&ent(6)));
    }

    #[test]
    fn caps_at_max_entries_per_segment() {
        // One huge turn — capped so the spine doesn't get one massive
        // marker.
        let entries: Vec<SegmentEntry> = (1..=10)
            .map(|i| entry(i as u8, &[1], i == 1))
            .collect();
        let opts = SegmentOpts {
            max_entries_per_segment: 4,
        };
        let out = segment(&entries, &opts);
        assert_eq!(out.iter().map(|s| s.entry_ids.len()).sum::<usize>(), 10);
        assert!(out.iter().all(|s| s.entry_ids.len() <= 4));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn entity_ranking_by_mention_count() {
        let entries = vec![
            entry(1, &[1, 2, 3], true),
            entry(2, &[1, 2, 4], false),
            entry(3, &[1, 2, 5], false),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        let seg = &out[0];
        // Counts: 1→3, 2→3, 3→1, 4→1, 5→1. Top two by count, then
        // first-seen tiebreak.
        assert_eq!(seg.entity_ids[0], ent(1));
        assert_eq!(seg.entity_ids[1], ent(2));
        assert_eq!(seg.entity_ids[2..], [ent(3), ent(4), ent(5)]);
    }

    #[test]
    fn entry_first_and_last_match_range() {
        let entries = vec![
            entry(1, &[1, 2, 3], true),
            entry(2, &[1, 2, 3], false),
            entry(3, &[1, 2, 3], false),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_first, entries[0].entry_id);
        assert_eq!(out[0].entry_last, entries[2].entry_id);
    }

    #[test]
    fn entries_before_first_turn_start_get_their_own_segment() {
        // Defensive: a leading entry that's not a turn-start (rare —
        // typically a system banner) collects until the first user
        // message arrives.
        let entries = vec![
            entry(1, &[9], false),    // no turn-start yet
            entry(2, &[1, 2], true),  // turn 1 user
            entry(3, &[1, 3], false), // turn 1 assistant
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].entry_ids.len(), 1);
        assert_eq!(out[1].entry_ids.len(), 2);
    }
}
