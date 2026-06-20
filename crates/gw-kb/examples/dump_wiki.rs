//! One-off verification utility for the wiki provenance pipeline.
//! Calls `fetch_wiki_doc` against a live KB and prints each section's
//! mentions with their section-local offsets, plus the surrounding
//! markdown slice — letting a human eyeball that surface text actually
//! lines up with the recorded byte range.
//!
//! Usage:
//!   DATABASE_URL=postgres://gw:gw@localhost:5432/greatwheel \
//!     cargo run -p gw-kb --example dump_wiki -- <source_uuid>

use std::env;

use gw_kb::wiki::fetch_wiki_doc;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let source_id_arg = args
        .get(1)
        .ok_or("usage: dump_wiki <source_uuid>")?;
    let source_id = Uuid::parse_str(source_id_arg)?;

    let url = env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://gw:gw@localhost:5432/greatwheel".into());
    let pool = PgPoolOptions::new().max_connections(2).connect(&url).await?;

    let doc = fetch_wiki_doc(&pool, source_id).await?;

    println!("Title: {}", doc.source.title);
    println!("Sections: {}", doc.sections.len());
    let total_mentions: usize = doc.sections.iter().map(|s| s.mentions.len()).sum();
    println!("Total spanned mentions across sections: {}", total_mentions);

    for sec in doc.sections.iter().take(5) {
        if sec.mentions.is_empty() {
            continue;
        }
        println!("\n── Section: {} ─────────────────", sec.anchor);
        println!(
            "heading_path: {:?}, markdown_len: {}, mentions: {}",
            sec.heading_path,
            sec.markdown.len(),
            sec.mentions.len()
        );
        for m in &sec.mentions {
            let slice = sec
                .markdown
                .get(m.norm_start..m.norm_end)
                .unwrap_or("<BAD OFFSETS>");
            let ok = slice == m.surface;
            println!(
                "  [{:>5}..{:<5}] kind={} label={:?} surface={:?}  markdown_slice={:?}  match={}",
                m.norm_start, m.norm_end, m.kind, m.label, m.surface, slice, ok
            );
        }
    }

    println!("\n--- Sidebar entity rollup ---");
    for e in doc.entities.iter().take(10) {
        println!(
            "  {:>5} mentions  [{}] {}",
            e.mentions_in_doc, e.kind, e.label
        );
    }

    Ok(())
}
