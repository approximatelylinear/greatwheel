-- Phase 1 sessions API: titles, summaries, archive flag, and a
-- per-user-active index for the sidebar listing.
--
-- title: short label shown in the session sidebar. NULL until the
--   auto-titler fills it from the first user message; clients can
--   PATCH it later to rename. Bounded only by the column type;
--   deliberately TEXT (not VARCHAR(N)) since titles are render-time
--   truncated by the frontend, not validated server-side.
--
-- summary: optional longer-form description. Reserved for a future
--   per-session summary — generated either lazily on resume or as
--   part of compaction. Nullable; no caller writes it yet.
--
-- archived_at: NULL = active session; non-NULL = user archived it
--   (kept for history, hidden from the default listing). Mirrors the
--   committed_at convention used in session_segments.
--
-- The new index supports the sidebar's hot path: "newest non-archived
-- sessions for this user." Putting archived_at second in the column
-- list keeps NULL rows clustered and lets a `WHERE archived_at IS NULL`
-- predicate use the index without scanning archived ones.

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS sessions_user_active
    ON sessions (user_id, archived_at, last_active_at DESC);
