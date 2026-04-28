-- Spine Issue #2 step A: session_segments cache.
--
-- See `docs/design-semantic-spine.md` §3.3. A segment is a contiguous
-- run of session entries that share ≥2 entities (or that the
-- centroid-cosine fallback grouped together). Segments are derivable
-- from session_entry_entities, but labelling is one LLM call per new
-- segment so we cache.
--
-- Re-segmentation marks old segments invalidated rather than deleting
-- them — history replay should show the labels that were live at the
-- time. The partial index makes the "current segments" query cheap.

CREATE TABLE session_segments (
    segment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,                            -- LLM-generated short name (1-3 words) or "N entities" fallback
    kind            TEXT NOT NULL,                            -- "comparison" | "decision" | "deep_dive" | "construction" | "other"
    entry_first     UUID NOT NULL REFERENCES session_entries(id) ON DELETE CASCADE,
    entry_last      UUID NOT NULL REFERENCES session_entries(id) ON DELETE CASCADE,
    -- Top entities in the segment, ranked by mention count. Capped
    -- by the segmentation pass before insert so the array stays
    -- bounded (~10 elements).
    entity_ids      UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    summary         TEXT,                                     -- optional 1-2 sentence LLM summary
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Set when re-segmentation supersedes this segment. NULL means
    -- "current" — see the partial index below.
    invalidated_at  TIMESTAMPTZ
);

-- "Current segments for this session" is the hot-path query (every
-- spine render runs it). Partial index keeps it small even on a
-- corpus where re-segmentation has run many times.
CREATE INDEX idx_session_segments_current
    ON session_segments (session_id, entry_first)
    WHERE invalidated_at IS NULL;

CREATE INDEX idx_session_segments_kind ON session_segments (kind);
