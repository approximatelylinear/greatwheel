//! Segmentation: contiguous-entry clustering by shared entities.
//!
//! The pure half of design-semantic-spine.md §4.2 — given a list of
//! entries with their entity-id lists, group adjacent entries into
//! segments where each entry shares ≥`min_shared_entities` with the
//! growing segment's entity set, capped at `max_entries_per_segment`.
//!
//! No DB, no LLM, no embedder dependency: the algorithm just operates
//! on `Vec<Uuid>` per entry. The orchestrator (step B) loads entries +
//! entity links from Postgres, calls this, then labels new segments
//! via LLM and diffs against the cache.
//!
//! ### Why not the cosine fallback yet?
//!
//! The design doc allows extending a segment when a new entry's
//! entity-vector centroid has cosine ≥ 0.6 with the segment's
//! centroid. That's strictly more permissive than the count-based
//! check and would catch e.g. "BM25 → ColBERT" → "ColBERT →
//! cross-encoder rerank" (which share only 1 entity directly but are
//! topically tight). Implementing it requires loading
//! `kb_entities.vector` for each entity, which makes the algorithm
//! DB-touching and turns a unit test into an integration test. Punt
//! to a follow-up; the count-based heuristic alone produces sensible
//! groupings on the corpora we have.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

use gw_core::EntryId;

/// Default minimum shared entities required to extend a segment with
/// the next entry. Matches design-semantic-spine.md §4.2.
pub const DEFAULT_MIN_SHARED_ENTITIES: usize = 2;
/// Default cap on segment length, so one entity-rich span doesn't
/// monopolise the whole spine.
pub const DEFAULT_MAX_ENTRIES_PER_SEGMENT: usize = 8;

/// Tunable knobs for the segmenter.
#[derive(Debug, Clone, Copy)]
pub struct SegmentOpts {
    pub min_shared_entities: usize,
    pub max_entries_per_segment: usize,
}

impl Default for SegmentOpts {
    fn default() -> Self {
        Self {
            min_shared_entities: DEFAULT_MIN_SHARED_ENTITIES,
            max_entries_per_segment: DEFAULT_MAX_ENTRIES_PER_SEGMENT,
        }
    }
}

/// One entry's input to segmentation: its id and the canonical
/// entity_ids appearing in it (already canonicalised through
/// `kb_entities` by the spine extractor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentEntry {
    pub entry_id: EntryId,
    pub entity_ids: Vec<Uuid>,
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

/// Segment a sequence of entries by shared-entity contiguity. Returns
/// segments in entry order. Empty input → empty output.
///
/// Algorithm (greedy single-pass):
///   1. The first entry seeds segment 0.
///   2. For each subsequent entry, count overlap with the current
///      segment's entity set. If overlap ≥ `min_shared_entities`
///      AND the segment hasn't hit `max_entries_per_segment`, append.
///      Otherwise close the segment and start a new one.
///   3. Singleton entries (no entity overlap with neighbour) become
///      their own segment. The labeller's fallback ("N entities")
///      handles those gracefully.
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
    // (count, first_seen_seq) for ordering: count desc, then
    // first-seen ascending so ties are stable across runs.
    let mut seq: u32 = 0;

    for e in entries {
        let entry_set: HashSet<Uuid> = e.entity_ids.iter().copied().collect();

        let can_extend = !cur_entry_ids.is_empty()
            && cur_entry_ids.len() < opts.max_entries_per_segment
            && entry_set.intersection(&cur_set).count() >= opts.min_shared_entities;

        if !can_extend && !cur_entry_ids.is_empty() {
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

    fn entry(id: u8, ents: &[usize]) -> SegmentEntry {
        let mut bytes = [0u8; 16];
        bytes[0] = id;
        SegmentEntry {
            entry_id: EntryId(Uuid::from_bytes(bytes)),
            entity_ids: ents.iter().map(|n| ent(*n)).collect(),
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let out = segment(&[], &SegmentOpts::default());
        assert!(out.is_empty());
    }

    #[test]
    fn single_entry_makes_singleton_segment() {
        let entries = vec![entry(1, &[1, 2, 3])];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_ids.len(), 1);
        // Top entities = all three, in first-seen order (counts equal).
        assert_eq!(out[0].entity_ids.len(), 3);
    }

    #[test]
    fn merges_adjacent_when_two_entities_shared() {
        let entries = vec![
            entry(1, &[1, 2, 3]),
            // shares {2, 3} with previous → 2 overlap → extend
            entry(2, &[2, 3, 4]),
            // shares {3, 4} with current segment {1,2,3,4} → still
            // ≥2 overlap → extend
            entry(3, &[3, 4, 5]),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_ids.len(), 3);
    }

    #[test]
    fn breaks_when_only_one_entity_shared() {
        let entries = vec![
            entry(1, &[1, 2, 3]),
            // shares {3} with previous → 1 overlap → break
            entry(2, &[3, 4, 5]),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].entry_ids.len(), 1);
        assert_eq!(out[1].entry_ids.len(), 1);
    }

    #[test]
    fn breaks_when_no_entities_shared() {
        let entries = vec![entry(1, &[1, 2, 3]), entry(2, &[4, 5, 6])];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn caps_at_max_entries_per_segment() {
        // Every entry shares 2 entities with the next, so without a
        // cap they'd all merge into one giant segment.
        let entries: Vec<SegmentEntry> = (1..=10)
            .map(|i| entry(i as u8, &[1, 2, 3]))
            .collect();
        let opts = SegmentOpts {
            max_entries_per_segment: 4,
            ..Default::default()
        };
        let out = segment(&entries, &opts);
        // 10 / 4 = 2 full + 1 remainder of 2
        assert_eq!(out.iter().map(|s| s.entry_ids.len()).sum::<usize>(), 10);
        assert!(out.iter().all(|s| s.entry_ids.len() <= 4));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn entity_ranking_by_mention_count() {
        let entries = vec![
            entry(1, &[1, 2, 3]),
            entry(2, &[1, 2, 4]),
            entry(3, &[1, 2, 5]),
        ];
        // ent(1) and ent(2) appear in all 3 (tied top), ent(3,4,5) in 1.
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        let seg = &out[0];
        // Top two: counts equal (3), tiebreaker is first-seen → ent(1), ent(2).
        assert_eq!(seg.entity_ids[0], ent(1));
        assert_eq!(seg.entity_ids[1], ent(2));
        // Remaining three: counts equal (1), first-seen order → ent(3), ent(4), ent(5).
        assert_eq!(seg.entity_ids[2..], [ent(3), ent(4), ent(5)]);
    }

    #[test]
    fn entry_first_and_last_match_range() {
        let entries = vec![
            entry(1, &[1, 2, 3]),
            entry(2, &[1, 2, 3]),
            entry(3, &[1, 2, 3]),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_first, entries[0].entry_id);
        assert_eq!(out[0].entry_last, entries[2].entry_id);
    }

    #[test]
    fn singleton_with_no_entities_starts_fresh_segment() {
        // An entry with empty entity list can never satisfy
        // min_shared_entities ≥ 2 — it always lands in its own segment.
        let entries = vec![
            entry(1, &[1, 2]),
            entry(2, &[]),
            entry(3, &[1, 2]),
        ];
        let out = segment(&entries, &SegmentOpts::default());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn min_shared_one_relaxes_grouping() {
        let entries = vec![
            entry(1, &[1, 2, 3]),
            entry(2, &[3, 4, 5]),
        ];
        let opts = SegmentOpts {
            min_shared_entities: 1,
            ..Default::default()
        };
        let out = segment(&entries, &opts);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].entry_ids.len(), 2);
    }
}
