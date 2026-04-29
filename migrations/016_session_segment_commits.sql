-- Spine Issue #5: commit/uncommit segments to a workspace.
--
-- "Committed" segments are user-curated keepers — the threads worth
-- coming back to. They appear in the workspace view (a separate
-- panel, not the live rail) so a long exploration session can be
-- distilled into a curated reading list.
--
-- Single column on session_segments rather than a separate table:
--   - one segment, one commit state — no need for history
--   - travels with segment_id; if a future resegment creates a new
--     id the commit is dropped (acceptable for v1; carry-forward
--     can be added later if it becomes a pain point)
--   - cheap "is committed?" check on the existing row, no join
--
-- NULL = transient (default); non-NULL = saved to workspace at this
-- timestamp. Even invalidated segments can stay committed — the
-- workspace shows them with a "(superseded)" tag so the user can
-- still revisit history.

ALTER TABLE session_segments
    ADD COLUMN committed_at TIMESTAMPTZ;

-- Workspace listing query: "all committed segments for this session,
-- newest commit first." The partial index keeps it small — we only
-- expect a handful of committed segments per session.
CREATE INDEX idx_session_segments_committed
    ON session_segments (session_id, committed_at DESC)
    WHERE committed_at IS NOT NULL;
