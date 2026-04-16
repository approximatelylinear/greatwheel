//! RSS/Atom feed subscriptions.
//!
//! A feed is a registered URL whose entries get ingested automatically
//! by `gw-kb feed sync`. Each entry becomes a source via the normal
//! `ingest_url` pipeline, with its `feed_id` stamped so we can trace
//! provenance.
//!
//! ### Sync semantics
//!
//! For each feed we fetch the raw XML, parse it with `feed-rs`
//! (handles RSS 0.9/1.0/2.0 and Atom 0.3/1.0), and for each entry with
//! a link we:
//!
//!   1. Check if the URL is already in `kb_sources`. If yes, skip —
//!      avoids re-fetching the full article for stuff we've already
//!      ingested.
//!   2. Otherwise call `ingest_url` (the normal pipeline). On success,
//!      stamp the resulting source row with `feed_id`.
//!   3. Track the newest entry publication time per feed so we can
//!      surface "what's new since last sync" in a future delta command.
//!
//! Errors per entry are logged and counted but don't abort the sync —
//! one 404 shouldn't kill the whole feed refresh.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::{ingest_url, KbStores};
use crate::topics::slugify;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feed {
    pub feed_id: Uuid,
    pub name: String,
    pub slug: String,
    pub url: String,
    pub feed_format: Option<String>,
    pub last_synced_at: Option<DateTime<Utc>>,
    pub last_entry_seen_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct FeedSyncReport {
    pub feed_name: String,
    pub entries_seen: usize,
    pub entries_ingested: usize,
    pub entries_skipped_existing: usize,
    pub entries_without_link: usize,
    pub entries_failed: usize,
}

/// Register a new feed. If `name` is `None`, derive it from the URL host.
pub async fn add_feed(pool: &PgPool, url: &str, name: Option<&str>) -> Result<Feed, KbError> {
    let url_trimmed = url.trim();
    if url_trimmed.is_empty() {
        return Err(KbError::Other("feed url is empty".into()));
    }

    let name = name
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| derive_name_from_url(url_trimmed));

    let base_slug = slugify(&name);
    let slug = unique_slug(pool, &base_slug).await?;

    let feed_id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO kb_feeds (name, slug, url)
        VALUES ($1, $2, $3)
        RETURNING feed_id
        "#,
    )
    .bind(&name)
    .bind(&slug)
    .bind(url_trimmed)
    .fetch_one(pool)
    .await?;

    fetch_feed_by_id(pool, feed_id).await
}

async fn unique_slug(pool: &PgPool, base: &str) -> Result<String, KbError> {
    // Cheap uniqueness loop — at most a handful of tries for typical feeds.
    let mut candidate = base.to_string();
    let mut n = 2;
    loop {
        let exists: Option<Uuid> =
            sqlx::query_scalar("SELECT feed_id FROM kb_feeds WHERE slug = $1")
                .bind(&candidate)
                .fetch_optional(pool)
                .await?;
        if exists.is_none() {
            return Ok(candidate);
        }
        candidate = format!("{}-{}", base, n);
        n += 1;
        if n > 100 {
            return Err(KbError::Other(format!(
                "could not find unique slug for {base}"
            )));
        }
    }
}

fn derive_name_from_url(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()))
        .unwrap_or_else(|| "feed".to_string())
}

/// List all feeds, newest-added first.
pub async fn list_feeds(pool: &PgPool) -> Result<Vec<Feed>, KbError> {
    let rows: Vec<FeedRow> = sqlx::query_as(
        r#"
        SELECT feed_id, name, slug, url, feed_format,
               last_synced_at, last_entry_seen_at,
               created_at, updated_at
        FROM kb_feeds
        ORDER BY created_at DESC
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(Feed::from).collect())
}

/// Remove a feed by slug. `kb_sources.feed_id` columns pointing at it
/// get set to NULL via the schema's ON DELETE SET NULL.
pub async fn remove_feed(pool: &PgPool, slug: &str) -> Result<bool, KbError> {
    let rows = sqlx::query("DELETE FROM kb_feeds WHERE slug = $1")
        .bind(slug)
        .execute(pool)
        .await?;
    Ok(rows.rows_affected() > 0)
}

async fn fetch_feed_by_id(pool: &PgPool, feed_id: Uuid) -> Result<Feed, KbError> {
    let row: FeedRow = sqlx::query_as(
        r#"
        SELECT feed_id, name, slug, url, feed_format,
               last_synced_at, last_entry_seen_at,
               created_at, updated_at
        FROM kb_feeds
        WHERE feed_id = $1
        "#,
    )
    .bind(feed_id)
    .fetch_one(pool)
    .await?;
    Ok(Feed::from(row))
}

pub async fn fetch_feed_by_slug(pool: &PgPool, slug: &str) -> Result<Feed, KbError> {
    let row: Option<FeedRow> = sqlx::query_as(
        r#"
        SELECT feed_id, name, slug, url, feed_format,
               last_synced_at, last_entry_seen_at,
               created_at, updated_at
        FROM kb_feeds
        WHERE slug = $1
        "#,
    )
    .bind(slug)
    .fetch_optional(pool)
    .await?;
    row.map(Feed::from)
        .ok_or_else(|| KbError::Other(format!("no feed with slug '{slug}'")))
}

/// Sync every registered feed.
pub async fn sync_all(
    stores: &KbStores,
    per_feed_limit: Option<usize>,
) -> Result<Vec<FeedSyncReport>, KbError> {
    let feeds = list_feeds(&stores.pg).await?;
    if feeds.is_empty() {
        return Ok(Vec::new());
    }
    let mut reports = Vec::with_capacity(feeds.len());
    for feed in feeds {
        match sync_one(stores, &feed, per_feed_limit).await {
            Ok(r) => reports.push(r),
            Err(e) => {
                warn!(feed = %feed.slug, err = %e, "feed sync failed");
                reports.push(FeedSyncReport {
                    feed_name: feed.name,
                    entries_failed: 1,
                    ..Default::default()
                });
            }
        }
    }
    Ok(reports)
}

/// Sync one feed by slug.
pub async fn sync_by_slug(
    stores: &KbStores,
    slug: &str,
    limit: Option<usize>,
) -> Result<FeedSyncReport, KbError> {
    let feed = fetch_feed_by_slug(&stores.pg, slug).await?;
    sync_one(stores, &feed, limit).await
}

async fn sync_one(
    stores: &KbStores,
    feed: &Feed,
    limit: Option<usize>,
) -> Result<FeedSyncReport, KbError> {
    info!(slug = %feed.slug, url = %feed.url, "syncing feed");
    let mut report = FeedSyncReport {
        feed_name: feed.name.clone(),
        ..Default::default()
    };

    // 1. Fetch + parse
    let entries = fetch_and_parse(&feed.url).await?;
    info!(slug = %feed.slug, entries = entries.len(), "parsed feed");

    // 2. Walk entries in order, ingest new ones
    let max = limit.unwrap_or(usize::MAX);
    let mut newest_seen: Option<DateTime<Utc>> = None;

    for entry in entries.into_iter().take(max) {
        report.entries_seen += 1;

        let Some(link) = entry.link.clone() else {
            report.entries_without_link += 1;
            continue;
        };
        if let Some(published) = entry.published {
            if newest_seen.is_none_or(|n| published > n) {
                newest_seen = Some(published);
            }
        }

        // Skip entries whose URL is already in kb_sources — avoids
        // a full HTTP fetch + extract for stuff we've seen before.
        // Check both the raw feed link and the rewritten form (ingest_url
        // normalises arxiv /abs/ → /pdf/), so previously-ingested arxiv
        // papers don't get re-ingested just because the feed surfaces
        // the /abs/ URL.
        let rewritten = crate::ingest::rewrite_url_for_ingest_public(&link);
        let exists: Option<Uuid> = sqlx::query_scalar(
            "SELECT source_id FROM kb_sources WHERE url = $1 OR url = $2 LIMIT 1",
        )
        .bind(&link)
        .bind(&rewritten)
        .fetch_optional(&stores.pg)
        .await?;
        if exists.is_some() {
            report.entries_skipped_existing += 1;
            continue;
        }

        // 3. Full ingest via the normal pipeline
        info!(entry = %entry.title.as_deref().unwrap_or("(untitled)"), link = %link, "ingesting feed entry");
        let ingest_report = match ingest_url(stores, &link).await {
            Ok(r) => r,
            Err(e) => {
                warn!(err = %e, link = %link, "feed entry ingest failed");
                report.entries_failed += 1;
                continue;
            }
        };

        // 4. Stamp feed_id on the resulting source
        sqlx::query("UPDATE kb_sources SET feed_id = $1 WHERE source_id = $2")
            .bind(feed.feed_id)
            .bind(ingest_report.source.source_id)
            .execute(&stores.pg)
            .await?;

        report.entries_ingested += 1;
    }

    // 5. Update feed cursors
    sqlx::query(
        r#"
        UPDATE kb_feeds
        SET last_synced_at = now(),
            last_entry_seen_at = COALESCE($1, last_entry_seen_at),
            updated_at = now()
        WHERE feed_id = $2
        "#,
    )
    .bind(newest_seen)
    .bind(feed.feed_id)
    .execute(&stores.pg)
    .await?;

    info!(
        slug = %feed.slug,
        ingested = report.entries_ingested,
        skipped = report.entries_skipped_existing,
        failed = report.entries_failed,
        "feed sync done"
    );
    Ok(report)
}

/// A minimal projection of a feed-rs entry with the fields we actually use.
#[derive(Debug, Clone)]
struct ParsedEntry {
    title: Option<String>,
    link: Option<String>,
    published: Option<DateTime<Utc>>,
}

async fn fetch_and_parse(url: &str) -> Result<Vec<ParsedEntry>, KbError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("gw-kb/0.1 feed sync")
        .build()
        .map_err(|e| KbError::Other(format!("http client: {e}")))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| KbError::Other(format!("fetch {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(KbError::Other(format!("fetch {url}: HTTP {status}")));
    }
    let body = resp
        .bytes()
        .await
        .map_err(|e| KbError::Other(format!("read {url}: {e}")))?;

    let feed = feed_rs::parser::parse(body.as_ref())
        .map_err(|e| KbError::Other(format!("feed parse {url}: {e}")))?;

    let out: Vec<ParsedEntry> = feed
        .entries
        .into_iter()
        .map(|entry| {
            let link = entry
                .links
                .into_iter()
                .find(|l| !l.href.is_empty())
                .map(|l| l.href);
            let title = entry.title.map(|t| t.content);
            let published = entry.published.or(entry.updated);
            ParsedEntry {
                title,
                link,
                published,
            }
        })
        .collect();
    Ok(out)
}

// --- sqlx glue ---

#[derive(sqlx::FromRow)]
struct FeedRow {
    feed_id: Uuid,
    name: String,
    slug: String,
    url: String,
    feed_format: Option<String>,
    last_synced_at: Option<DateTime<Utc>>,
    last_entry_seen_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<FeedRow> for Feed {
    fn from(r: FeedRow) -> Self {
        Feed {
            feed_id: r.feed_id,
            name: r.name,
            slug: r.slug,
            url: r.url,
            feed_format: r.feed_format,
            last_synced_at: r.last_synced_at,
            last_entry_seen_at: r.last_entry_seen_at,
            created_at: r.created_at,
            updated_at: r.updated_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_name_from_url_takes_host() {
        assert_eq!(
            derive_name_from_url("https://huggingface.co/blog/feed.xml"),
            "huggingface.co"
        );
        assert_eq!(
            derive_name_from_url("https://example.org/rss"),
            "example.org"
        );
        assert_eq!(derive_name_from_url("nonsense"), "feed");
    }
}
