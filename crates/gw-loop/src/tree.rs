use chrono::Utc;
use gw_core::{EntryId, EntryType, SessionEntry, SessionId};

use crate::pg_store::PgSessionStore;

/// In-memory session tree with optional Postgres write-through.
///
/// The in-memory Vec is the source of truth during an active session.
/// If a PgSessionStore is attached, new entries are flushed to Postgres
/// after each turn via `flush_to_pg()`.
pub struct SessionTree {
    session_id: SessionId,
    entries: Vec<SessionEntry>,
    active_leaf: Option<EntryId>,
    /// Index of the first entry not yet flushed to Postgres.
    flush_cursor: usize,
    /// Optional Postgres store for durability.
    pg_store: Option<PgSessionStore>,
}

impl SessionTree {
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            entries: Vec::new(),
            active_leaf: None,
            flush_cursor: 0,
            pg_store: None,
        }
    }

    /// Create a tree with Postgres backing.
    pub fn with_pg(session_id: SessionId, pg_store: PgSessionStore) -> Self {
        Self {
            session_id,
            entries: Vec::new(),
            active_leaf: None,
            flush_cursor: 0,
            pg_store: Some(pg_store),
        }
    }

    /// Load a tree from Postgres, restoring all entries and active leaf.
    pub async fn load_from_pg(
        session_id: SessionId,
        pg_store: PgSessionStore,
    ) -> Result<Self, sqlx::Error> {
        let entries = pg_store.load_entries(session_id).await?;
        let active_leaf = pg_store.load_active_leaf(session_id).await?;
        let flush_cursor = entries.len(); // All loaded entries are already persisted.

        Ok(Self {
            session_id,
            entries,
            active_leaf,
            flush_cursor,
            pg_store: Some(pg_store),
        })
    }

    /// Flush unflushed entries to Postgres and update the active leaf.
    /// Returns the slice of entries newly written by this call (empty
    /// when nothing was unflushed). Callers can use the returned
    /// slice to fan out background work that needs the entries to
    /// already be present in Postgres — e.g. the spine extractor's
    /// FK-bound writes against `session_entries(id)`.
    /// No-op + empty Vec if no PgSessionStore is attached.
    pub async fn flush_to_pg(&mut self) -> Result<Vec<SessionEntry>, sqlx::Error> {
        let pg_store = match &self.pg_store {
            Some(store) => store.clone(),
            None => return Ok(Vec::new()),
        };

        let new_entries: Vec<SessionEntry> = if self.flush_cursor < self.entries.len() {
            let entries = self.entries[self.flush_cursor..].to_vec();
            pg_store.insert_entries(&entries).await?;
            self.flush_cursor = self.entries.len();
            entries
        } else {
            Vec::new()
        };

        pg_store
            .update_active_leaf(self.session_id, self.active_leaf)
            .await?;

        Ok(new_entries)
    }

    /// Append an entry as a child of the current active leaf.
    pub fn append(&mut self, entry_type: EntryType) -> EntryId {
        let id = EntryId::new();
        let entry = SessionEntry {
            id,
            session_id: self.session_id,
            parent_id: self.active_leaf,
            entry_type,
            created_at: Utc::now(),
        };
        self.entries.push(entry);
        self.active_leaf = Some(id);
        id
    }

    /// Append an entry as a child of a specific parent.
    pub fn append_at(&mut self, parent: EntryId, entry_type: EntryType) -> EntryId {
        let id = EntryId::new();
        let entry = SessionEntry {
            id,
            session_id: self.session_id,
            parent_id: Some(parent),
            entry_type,
            created_at: Utc::now(),
        };
        self.entries.push(entry);
        self.active_leaf = Some(id);
        id
    }

    /// Walk from the active leaf to the root, returning entries in root-first order.
    pub fn path_to_leaf(&self) -> Vec<&SessionEntry> {
        let mut path = Vec::new();
        let mut current = self.active_leaf;
        while let Some(id) = current {
            if let Some(entry) = self.find_entry(id) {
                path.push(entry);
                current = entry.parent_id;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    pub fn set_active_leaf(&mut self, id: EntryId) {
        self.active_leaf = Some(id);
    }

    pub fn active_leaf(&self) -> Option<EntryId> {
        self.active_leaf
    }

    pub fn entries(&self) -> &[SessionEntry] {
        &self.entries
    }

    pub fn find_entry(&self, id: EntryId) -> Option<&SessionEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Whether this tree has a Postgres store attached.
    pub fn has_pg(&self) -> bool {
        self.pg_store.is_some()
    }

    /// Walk ancestors from `id` to root, returning the set of entry IDs on that path.
    fn ancestor_ids(&self, id: EntryId) -> Vec<EntryId> {
        let mut ids = Vec::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            ids.push(cid);
            current = self.find_entry(cid).and_then(|e| e.parent_id);
        }
        ids
    }

    /// Walk from root to `id`, returning entries in root-first order.
    pub fn path_to(&self, id: EntryId) -> Vec<&SessionEntry> {
        let mut path = Vec::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            if let Some(entry) = self.find_entry(cid) {
                path.push(entry);
                current = entry.parent_id;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Get entries on the branch from `from_leaf` that are NOT on the path to `to_target`.
    pub fn branch_entries(&self, from_leaf: EntryId, to_target: EntryId) -> Vec<&SessionEntry> {
        let target_ancestors: std::collections::HashSet<EntryId> =
            self.ancestor_ids(to_target).into_iter().collect();
        let from_path = self.path_to(from_leaf);
        from_path
            .into_iter()
            .filter(|e| !target_ancestors.contains(&e.id))
            .collect()
    }

    /// Find the most recent ReplSnapshot on the path to an entry.
    pub fn find_latest_snapshot(&self, leaf: EntryId) -> Option<&gw_core::ReplSnapshotData> {
        let path = self.path_to(leaf);
        for entry in path.iter().rev() {
            match &entry.entry_type {
                EntryType::ReplSnapshot(data) => return Some(data),
                EntryType::Compaction { snapshot, .. } => return Some(snapshot),
                _ => {}
            }
        }
        None
    }

    /// Count user message entries on the current path.
    pub fn user_turn_count(&self) -> usize {
        self.path_to_leaf()
            .iter()
            .filter(|e| matches!(e.entry_type, EntryType::UserMessage(_)))
            .count()
    }
}
