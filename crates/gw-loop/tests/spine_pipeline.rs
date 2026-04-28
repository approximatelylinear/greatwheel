//! Integration test for the semantic-spine pipeline.
//!
//! Drives 3 session entries through the post-LLM half of the spine
//! extractor (canonicalisation + relation resolution + persistence)
//! and asserts the rows land correctly. The LLM call is bypassed via
//! `SpineExtractor::extract_entry_from_raw`, which takes a hand-
//! crafted `RawJointExtraction` — that's the seam Issue #1 step E
//! exists for.
//!
//! Gated on a live dev environment (matching `gw-kb`'s
//! `agent_integration` test):
//!
//! ```bash
//! DATABASE_URL=postgres://gw:gw@localhost:5432/greatwheel \
//! PYO3_PYTHON=$HOME/Code/greatwheel/crates/gw-kb/python/.venv/bin/python \
//! GW_KB_PYTHON_PATH=$HOME/Code/greatwheel/crates/gw-kb/python \
//! GW_KB_EMBED_DEVICE=cpu \
//! cargo test -p gw-loop --test spine_pipeline -- --nocapture
//! ```
//!
//! The test cleans up after itself by deleting the session row it
//! inserted (which cascades to session_entries, session_entry_entities,
//! session_entry_relations via the FKs in migration 014).

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use gw_core::EntryId;
use gw_kb::embed::Embedder;
use gw_kb::index::{KbLanceStore, KbTantivyStore};
use gw_kb::ingest::KbStores;
use gw_llm::OllamaClient;
use gw_loop::spine::{persist::persist_entry_extraction, SpineExtractor};
use gw_loop::spine::types::{RawEntryEntity, RawJointExtraction, RawRelation};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use uuid::Uuid;

const EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";
const EMBEDDING_DIM: i32 = 768;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(manifest);
    workspace.join(rel)
}

fn preconditions_met() -> Option<(String, std::path::PathBuf, std::path::PathBuf)> {
    let Ok(pg_url) = std::env::var("DATABASE_URL") else {
        eprintln!("skip: DATABASE_URL not set");
        return None;
    };
    if std::env::var("PYO3_PYTHON").is_err() {
        eprintln!("skip: PYO3_PYTHON not set (needed for sentence-transformers)");
        return None;
    }
    if std::env::var("GW_KB_PYTHON_PATH").is_err() {
        eprintln!("skip: GW_KB_PYTHON_PATH not set");
        return None;
    }
    let lance = workspace_path("data/kb-lancedb");
    if !lance.exists() {
        eprintln!("skip: {} does not exist", lance.display());
        return None;
    }
    let tantivy = workspace_path("data/kb-tantivy");
    if !tantivy.exists() {
        eprintln!("skip: {} does not exist", tantivy.display());
        return None;
    }
    Some((pg_url, lance, tantivy))
}

async fn build_stores(pg_url: &str, lance_path: &Path, tantivy_path: &Path) -> KbStores {
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(pg_url)
        .await
        .expect("connect postgres");
    // Apply pending migrations (014_spine_entry_extraction.sql in
    // particular). Idempotent via the _sqlx_migrations tracking
    // table — same call literature_assistant + the gw-kb CLI make
    // on startup.
    sqlx::migrate!("../../migrations")
        .run(&pool)
        .await
        .expect("apply migrations");
    let lance = Arc::new(
        KbLanceStore::open(
            lance_path.to_str().expect("lance path utf8"),
            EMBEDDING_DIM,
        )
        .await
        .expect("open lance store"),
    );
    let tantivy = Arc::new(KbTantivyStore::open(tantivy_path).expect("open tantivy"));
    let embedder = Arc::new(Embedder::new(EMBEDDING_MODEL.to_string()));
    let llm = Arc::new(OllamaClient::new(
        "http://localhost:11434".to_string(),
        "http://localhost:11434".to_string(),
        "qwen3.5:9b".to_string(),
        "unused".to_string(),
    ));
    KbStores {
        pg: pool,
        lance,
        tantivy,
        embedder,
        llm,
    }
}

/// Bag of ids the test seeds + tears down. Sessions FK back through
/// agent_defs → orgs and users → orgs, so we have to plant the full
/// chain even though we only care about the session.
struct TestSeed {
    org_id: Uuid,
    user_id: Uuid,
    agent_id: Uuid,
    session_id: Uuid,
    entry_ids: [EntryId; 3],
}

/// Plant org → user → agent_def → session → 3 session_entries.
async fn seed_session(pg: &PgPool) -> TestSeed {
    let suffix = Uuid::new_v4().simple().to_string();

    let org_id: Uuid =
        sqlx::query_scalar("INSERT INTO orgs (name) VALUES ($1) RETURNING id")
            .bind(format!("spine-test-{suffix}"))
            .fetch_one(pg)
            .await
            .expect("insert org");

    let user_id: Uuid = sqlx::query_scalar(
        "INSERT INTO users (org_id, name, email) VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(org_id)
    .bind("spine-test")
    .bind(format!("spine-{suffix}@test.local"))
    .fetch_one(pg)
    .await
    .expect("insert user");

    let agent_id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO agent_defs
            (org_id, name, system_prompt, source_type,
             tool_permissions, model_config, resource_limits)
        VALUES ($1, $2, '', 'inline', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb)
        RETURNING id
        "#,
    )
    .bind(org_id)
    .bind(format!("spine-test-{suffix}"))
    .fetch_one(pg)
    .await
    .expect("insert agent_def");

    let session_id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO sessions (org_id, user_id, agent_id, session_key)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        "#,
    )
    .bind(org_id)
    .bind(user_id)
    .bind(agent_id)
    .bind(format!("spine-test-{suffix}"))
    .fetch_one(pg)
    .await
    .expect("insert session");

    let entry_ids: [EntryId; 3] = [EntryId::new(), EntryId::new(), EntryId::new()];
    let prose: [&str; 3] = [
        "Let's compare BM25 vs ColBERT for retrieval recall.",
        "Run ColBERT then a cross-encoder rerank — that's the pipeline.",
        "Decided: ColBERT top-200 with the rerank stage on top.",
    ];
    for (i, eid) in entry_ids.iter().enumerate() {
        sqlx::query(
            r#"
            INSERT INTO session_entries (id, session_id, parent_id, entry_type, content, created_at)
            VALUES ($1, $2, $3, 'UserMessage', $4, $5)
            "#,
        )
        .bind(eid.0)
        .bind(session_id)
        .bind::<Option<Uuid>>(if i == 0 {
            None
        } else {
            Some(entry_ids[i - 1].0)
        })
        .bind(serde_json::json!({ "UserMessage": prose[i] }))
        .bind(Utc::now())
        .execute(pg)
        .await
        .expect("insert session_entry");
    }

    TestSeed {
        org_id,
        user_id,
        agent_id,
        session_id,
        entry_ids,
    }
}

async fn cleanup(pg: &PgPool, seed: &TestSeed) {
    // Order matters — children before parents, even with cascades, so
    // a partial failure during the test leaves the DB in a clean
    // state. session_entries cascades to session_entry_entities and
    // session_entry_relations via the FKs in migration 014.
    let _ = sqlx::query("UPDATE sessions SET active_leaf_id = NULL WHERE id = $1")
        .bind(seed.session_id)
        .execute(pg)
        .await;
    let _ = sqlx::query("DELETE FROM session_entries WHERE session_id = $1")
        .bind(seed.session_id)
        .execute(pg)
        .await;
    let _ = sqlx::query("DELETE FROM sessions WHERE id = $1")
        .bind(seed.session_id)
        .execute(pg)
        .await;
    let _ = sqlx::query("DELETE FROM agent_defs WHERE id = $1")
        .bind(seed.agent_id)
        .execute(pg)
        .await;
    let _ = sqlx::query("DELETE FROM users WHERE id = $1")
        .bind(seed.user_id)
        .execute(pg)
        .await;
    let _ = sqlx::query("DELETE FROM orgs WHERE id = $1")
        .bind(seed.org_id)
        .execute(pg)
        .await;
}

fn ent(label: &str, kind: &str, role: &str) -> RawEntryEntity {
    RawEntryEntity {
        label: label.into(),
        kind: kind.into(),
        canonical_form: label.into(),
        role: role.into(),
        status: "mentioned".into(),
        confidence: 0.9,
        span_start: None,
        span_end: None,
    }
}

fn rel(subject: &str, object: &str, predicate: &str, directed: bool) -> RawRelation {
    RawRelation {
        subject: subject.into(),
        object: object.into(),
        predicate: predicate.into(),
        directed,
        surface: format!("{subject} {predicate} {object}"),
        confidence: 0.85,
        span_start: None,
        span_end: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spine_pipeline_persists_entities_and_relations() {
    let Some((pg_url, lance_path, tantivy_path)) = preconditions_met() else {
        return;
    };

    let stores = build_stores(&pg_url, &lance_path, &tantivy_path).await;
    let pg = stores.pg.clone();

    let seed = seed_session(&pg).await;
    let entry_ids = seed.entry_ids;
    let session_id = seed.session_id;
    let extractor = SpineExtractor::new(Arc::new(stores));

    // Hand-crafted joint extractions, one per entry — these stand
    // in for the LLM output that the bypassed call would produce.
    let extractions: [(EntryId, RawJointExtraction); 3] = [
        (
            entry_ids[0],
            RawJointExtraction {
                entities: vec![
                    ent("BM25", "method", "compared"),
                    ent("ColBERT", "method", "compared"),
                ],
                relations: vec![rel("BM25", "ColBERT", "compared_with", false)],
            },
        ),
        (
            entry_ids[1],
            RawJointExtraction {
                entities: vec![
                    ent("ColBERT", "method", "referenced"),
                    ent("cross-encoder rerank", "method", "referenced"),
                ],
                relations: vec![rel(
                    "ColBERT",
                    "cross-encoder rerank",
                    "composes",
                    true,
                )],
            },
        ),
        (
            entry_ids[2],
            RawJointExtraction {
                entities: vec![ent("ColBERT", "method", "decided")],
                relations: vec![],
            },
        ),
    ];

    let mut total_entity_links = 0usize;
    let mut total_relations = 0usize;
    for (entry_id, raw) in extractions {
        let extraction = extractor
            .extract_entry_from_raw(entry_id, raw)
            .await
            .expect("extract_entry_from_raw");
        let report = persist_entry_extraction(&pg, &extraction)
            .await
            .expect("persist_entry_extraction");
        total_entity_links += report.entity_links_written;
        total_relations += report.relations_written;
    }

    // Assertions ----------------------------------------------------

    // Five entity-mentions across the three entries: BM25 (e0),
    // ColBERT (e0, e1, e2), cross-encoder rerank (e1) = 5 rows.
    assert_eq!(
        total_entity_links, 5,
        "expected 5 session_entry_entities rows, got {total_entity_links}"
    );

    // Two relations: compared_with (e0), composes (e1).
    assert_eq!(
        total_relations, 2,
        "expected 2 session_entry_relations rows, got {total_relations}"
    );

    // Verify the rows are actually in the DB and tied to our session.
    let entry_count: (i64,) = sqlx::query_as(
        r#"
        SELECT COUNT(*) FROM session_entry_entities ee
        JOIN session_entries se ON se.id = ee.entry_id
        WHERE se.session_id = $1
        "#,
    )
    .bind(session_id)
    .fetch_one(&pg)
    .await
    .expect("count entry-entity rows");
    assert_eq!(entry_count.0, 5);

    let rel_count: (i64,) = sqlx::query_as(
        r#"
        SELECT COUNT(*) FROM session_entry_relations rr
        JOIN session_entries se ON se.id = rr.entry_id
        WHERE se.session_id = $1
        "#,
    )
    .bind(session_id)
    .fetch_one(&pg)
    .await
    .expect("count entry-relation rows");
    assert_eq!(rel_count.0, 2);

    // Canonicalisation should have collapsed three "ColBERT" mentions
    // (one per entry) into a single kb_entities row. We verify by
    // counting DISTINCT entity_ids attached to entries in this session.
    let distinct_entities: (i64,) = sqlx::query_as(
        r#"
        SELECT COUNT(DISTINCT ee.entity_id) FROM session_entry_entities ee
        JOIN session_entries se ON se.id = ee.entry_id
        WHERE se.session_id = $1
        "#,
    )
    .bind(session_id)
    .fetch_one(&pg)
    .await
    .expect("count distinct entities");
    assert_eq!(
        distinct_entities.0, 3,
        "expected 3 distinct canonical entities (BM25, ColBERT, cross-encoder rerank), got {}",
        distinct_entities.0
    );

    // Subject/object resolution: the compared_with relation should
    // point at BM25 and ColBERT (which are the two distinct method
    // entities mentioned in entry 0). Pull labels out of kb_entities
    // to make the assertion human-readable on failure.
    let compared_with_pair: (String, String) = sqlx::query_as(
        r#"
        SELECT s.label, o.label
        FROM session_entry_relations rr
        JOIN session_entries se ON se.id = rr.entry_id
        JOIN kb_entities s ON s.entity_id = rr.subject_id
        JOIN kb_entities o ON o.entity_id = rr.object_id
        WHERE se.session_id = $1
          AND rr.predicate = 'compared_with'
        "#,
    )
    .bind(session_id)
    .fetch_one(&pg)
    .await
    .expect("fetch compared_with");
    let (s_label, o_label) = compared_with_pair;
    let pair = {
        // symmetric — sort for deterministic compare
        let mut p = [s_label.clone(), o_label.clone()];
        p.sort();
        p
    };
    assert_eq!(
        pair,
        ["BM25".to_string(), "ColBERT".to_string()],
        "compared_with endpoints didn't resolve correctly"
    );

    cleanup(&pg, &seed).await;
}
