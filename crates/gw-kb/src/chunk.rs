//! Markdown-aware chunking.
//!
//! Strategy:
//!  1. Split on `## ` (H2) headings — natural topic boundaries
//!  2. If a section exceeds the target size, split on `### ` (H3)
//!  3. If still too large, split on paragraph (blank line) boundaries
//!  4. If still too large, split on sentence boundaries
//!
//! Each chunk retains its heading path so display surfaces have context.
//!
//! Token counting is approximate: we use `chars / 4` as a cheap proxy.
//! For our chunk sizes (~512 tokens) this is well within the noise of
//! actual tokenizer output, and avoids pulling in tokenizer dependencies.

use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;

/// Approximate target chunk size, in characters (~512 tokens).
pub const DEFAULT_TARGET_CHARS: usize = 2048;
/// Overlap between adjacent chunks, in characters (~64 tokens).
pub const DEFAULT_OVERLAP_CHARS: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub ordinal: usize,
    pub content: String,
    pub char_offset: usize,
    pub char_length: usize,
    pub heading_path: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkOpts {
    pub target_chars: usize,
    pub overlap_chars: usize,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        Self {
            target_chars: DEFAULT_TARGET_CHARS,
            overlap_chars: DEFAULT_OVERLAP_CHARS,
        }
    }
}

/// Split a markdown document into chunks following the strategy above.
pub fn chunk_markdown(markdown: &str, opts: ChunkOpts) -> Vec<Chunk> {
    let sections = split_by_headings(markdown);
    let mut chunks = Vec::new();
    let mut ordinal = 0;

    for section in sections {
        if section.text.chars().count() <= opts.target_chars {
            chunks.push(Chunk {
                ordinal,
                content: section.text,
                char_offset: section.offset,
                char_length: 0, // filled in below
                heading_path: section.heading_path,
            });
            ordinal += 1;
            continue;
        }

        // Section too large — split into sub-chunks at paragraph/sentence boundaries
        let sub = split_oversized(&section.text, section.offset, opts);
        for piece in sub {
            chunks.push(Chunk {
                ordinal,
                content: piece.0,
                char_offset: piece.1,
                char_length: 0,
                heading_path: section.heading_path.clone(),
            });
            ordinal += 1;
        }
    }

    // Fill in char_length now that content is finalized
    for c in &mut chunks {
        c.char_length = c.content.chars().count();
    }

    chunks
}

/// A markdown section delimited by headings.
struct Section {
    text: String,
    offset: usize, // character offset into the original markdown
    heading_path: Vec<String>,
}

/// Split a markdown document into sections at H1/H2/H3 boundaries while
/// tracking the active heading path. Each returned section's `offset` is
/// the absolute character index where the section begins in the source.
fn split_by_headings(markdown: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut current_text = String::new();
    let mut current_heading_path: Vec<String> = Vec::new();
    let mut char_offset = 0usize;
    let mut section_start = 0usize;

    let flush = |sections: &mut Vec<Section>, text: &mut String, start: usize, path: &[String]| {
        if !text.trim().is_empty() {
            sections.push(Section {
                text: std::mem::take(text),
                offset: start,
                heading_path: path.to_vec(),
            });
        }
    };

    for line in markdown.split_inclusive('\n') {
        let line_chars = line.chars().count();
        let trimmed = line.trim_start();

        let heading_level = if trimmed.starts_with("### ") {
            Some(3)
        } else if trimmed.starts_with("## ") {
            Some(2)
        } else if trimmed.starts_with("# ") {
            Some(1)
        } else {
            None
        };

        if let Some(level) = heading_level {
            // Flush the section we just finished, then start a new one
            // anchored at the heading line itself.
            flush(
                &mut sections,
                &mut current_text,
                section_start,
                &current_heading_path,
            );
            section_start = char_offset;

            let heading_text = trimmed.trim_end_matches('\n').to_string();
            update_heading_path(&mut current_heading_path, level, heading_text);
            current_text.push_str(line);
        } else {
            // First content line of a new section: anchor section_start here
            // (not at the previous flush point — there may be leading blanks)
            if current_text.is_empty() {
                section_start = char_offset;
            }
            current_text.push_str(line);
        }

        char_offset += line_chars;
    }

    flush(
        &mut sections,
        &mut current_text,
        section_start,
        &current_heading_path,
    );
    sections
}

/// Update the active heading chain when a new heading is encountered.
///
/// Semantics: `path` is the chain of headings leading to the current
/// content, ordered from outermost to innermost. When a heading appears
/// at level N, we drop any prior headings at level >= N (they're now
/// out of scope) and append the new one.
///
/// This avoids padding with empty strings when a deep heading appears
/// without a parent (e.g. a PDF that opens at H2 because the title lives
/// in PDF metadata, not the markdown body).
fn update_heading_path(path: &mut Vec<String>, level: usize, heading: String) {
    let target_depth = level.saturating_sub(1);
    if path.len() > target_depth {
        path.truncate(target_depth);
    }
    path.push(heading);
}

/// A character range within a section's `chars` vector. Atoms are the
/// smallest indivisible units the packer composes into chunks.
#[derive(Debug, Clone, Copy)]
struct Range {
    start: usize,
    end: usize,
}

impl Range {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Split an oversized section into chunks with correct char offsets.
///
/// Approach:
///   1. Atomize: walk paragraphs → sentences → hard-window slices, producing
///      a flat list of char ranges where each range is `<= target_chars`.
///   2. Pack: greedily concatenate adjacent atoms into chunks bounded by
///      `target_chars`.
///
/// Each output chunk is a slice of the original section text, so its
/// `char_offset` is `base_offset + atom.start`.
fn split_oversized(text: &str, base_offset: usize, opts: ChunkOpts) -> Vec<(String, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let atoms = atomize(&chars, opts.target_chars, opts.overlap_chars);
    pack_atoms(&chars, &atoms, base_offset, opts.target_chars)
}

fn atomize(chars: &[char], target: usize, overlap: usize) -> Vec<Range> {
    let mut atoms = Vec::new();
    for para in paragraph_ranges(chars) {
        if para.len() <= target {
            atoms.push(para);
            continue;
        }
        for sent in sentence_ranges(chars, para.start, para.end) {
            if sent.len() <= target {
                atoms.push(sent);
                continue;
            }
            // Last resort: hard window with overlap
            atoms.extend(hard_window_ranges(sent.start, sent.end, target, overlap));
        }
    }
    atoms
}

/// Greedily pack atoms into chunks. Each chunk is a slice
/// `chars[first_atom.start..last_atom.end]`. Atoms need not be contiguous —
/// the slice naturally captures any gap chars (e.g. paragraph-separating
/// `\n\n`) between them. The bound is on slice length, not atom-sum length.
fn pack_atoms(
    chars: &[char],
    atoms: &[Range],
    base_offset: usize,
    target: usize,
) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < atoms.len() {
        let chunk_start = atoms[i].start;
        let mut chunk_end = atoms[i].end;
        let mut j = i + 1;
        while j < atoms.len() {
            // Slice from chunk_start to atoms[j].end captures everything
            // (including separator whitespace) between atoms. Bound on the
            // slice length so chunks stay within target.
            let candidate_end = atoms[j].end;
            if candidate_end - chunk_start > target {
                break;
            }
            chunk_end = candidate_end;
            j += 1;
        }
        let text: String = chars[chunk_start..chunk_end].iter().collect();
        out.push((text, base_offset + chunk_start));
        i = j;
    }
    out
}

/// Find non-empty paragraph ranges in `chars`. Paragraphs are separated by
/// runs of `\n\n`. Range bounds are inclusive of trailing newlines that
/// belong to the paragraph and exclusive of the blank-line separator.
fn paragraph_ranges(chars: &[char]) -> Vec<Range> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut start = 0;

    while i < chars.len() {
        if chars[i] == '\n' && i + 1 < chars.len() && chars[i + 1] == '\n' {
            // Paragraph break — emit [start, i+1) so the trailing newline
            // belongs to the paragraph
            let end = i + 1;
            if start < end && chars[start..end].iter().any(|c| !c.is_whitespace()) {
                out.push(Range { start, end });
            }
            // Skip the run of newlines that separates paragraphs
            i += 1;
            while i < chars.len() && chars[i] == '\n' {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }

    if start < chars.len() && chars[start..].iter().any(|c| !c.is_whitespace()) {
        out.push(Range {
            start,
            end: chars.len(),
        });
    }
    out
}

/// Find sentence ranges within `chars[start..end]`. Splits on `.`, `!`, `?`
/// followed by whitespace and an uppercase letter. Falls back to a single
/// range if no sentence boundary is found.
fn sentence_ranges(chars: &[char], start: usize, end: usize) -> Vec<Range> {
    let mut out = Vec::new();
    let mut s = start;
    let mut i = start;

    while i < end {
        let is_terminal = matches!(chars[i], '.' | '!' | '?')
            && i + 1 < end
            && chars[i + 1].is_whitespace()
            && i + 2 < end
            && chars[i + 2].is_uppercase();

        if is_terminal {
            let sent_end = i + 1; // include the punctuation
            if s < sent_end {
                out.push(Range {
                    start: s,
                    end: sent_end,
                });
            }
            // Advance past the whitespace that separates sentences so the
            // next sentence's range starts on its first non-whitespace char.
            i = sent_end;
            while i < end && chars[i].is_whitespace() {
                i += 1;
            }
            s = i;
        } else {
            i += 1;
        }
    }

    if s < end {
        out.push(Range { start: s, end });
    }
    out
}

/// Hard window split as a list of ranges. Used as the last resort when
/// no structural boundary can break a sentence below `target`.
fn hard_window_ranges(start: usize, end: usize, target: usize, overlap: usize) -> Vec<Range> {
    let mut out = Vec::new();
    if end <= start {
        return out;
    }
    if end - start <= target {
        out.push(Range { start, end });
        return out;
    }
    let stride = target.saturating_sub(overlap).max(1);
    let mut s = start;
    while s < end {
        let e = (s + target).min(end);
        out.push(Range { start: s, end: e });
        if e == end {
            break;
        }
        s += stride;
    }
    out
}

/// Insert chunks for a source into Postgres. Returns the inserted chunk IDs
/// in the same order as the input.
pub async fn insert_chunks(
    pool: &PgPool,
    source_id: Uuid,
    chunks: &[Chunk],
) -> Result<Vec<Uuid>, KbError> {
    let mut ids = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let id: Uuid = sqlx::query_scalar(
            r#"
            INSERT INTO kb_chunks
                (source_id, ordinal, content, char_offset, char_length, heading_path)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING chunk_id
            "#,
        )
        .bind(source_id)
        .bind(chunk.ordinal as i32)
        .bind(&chunk.content)
        .bind(chunk.char_offset as i32)
        .bind(chunk.char_length as i32)
        .bind(&chunk.heading_path)
        .fetch_one(pool)
        .await?;
        ids.push(id);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_doc_single_chunk() {
        let md = "# Title\n\nA short paragraph.\n";
        let chunks = chunk_markdown(md, ChunkOpts::default());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("short paragraph"));
    }

    #[test]
    fn splits_on_h2() {
        let md = "# Doc\n\nintro\n\n## Section A\n\nbody A\n\n## Section B\n\nbody B\n";
        let chunks = chunk_markdown(md, ChunkOpts::default());
        assert!(chunks.len() >= 2);
        // Section A and B should be in different chunks with distinct heading paths
        let labels: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c.heading_path.last().map(|s| s.as_str()))
            .collect();
        assert!(labels.iter().any(|s| s.contains("Section A")));
        assert!(labels.iter().any(|s| s.contains("Section B")));
    }

    /// For every chunk produced from a document, the substring
    /// `doc[char_offset..char_offset+char_length]` (in chars, not bytes)
    /// must equal the chunk's content. This is the contract that
    /// downstream surfaces (highlighting, citation, re-fetch) rely on.
    fn assert_offsets_recoverable(doc: &str, chunks: &[Chunk]) {
        let doc_chars: Vec<char> = doc.chars().collect();
        for c in chunks {
            let end = c.char_offset + c.char_length;
            assert!(
                end <= doc_chars.len(),
                "chunk {} offset {} + length {} = {} > doc len {}",
                c.ordinal,
                c.char_offset,
                c.char_length,
                end,
                doc_chars.len()
            );
            let slice: String = doc_chars[c.char_offset..end].iter().collect();
            assert_eq!(
                slice, c.content,
                "chunk {} content does not match doc[{}..{}]",
                c.ordinal, c.char_offset, end
            );
        }
    }

    #[test]
    fn offsets_are_recoverable_small_doc() {
        let md = "# Title\n\nA short paragraph.\n\n## Section A\n\nbody A here.\n";
        let chunks = chunk_markdown(md, ChunkOpts::default());
        assert_offsets_recoverable(md, &chunks);
    }

    #[test]
    fn offsets_are_recoverable_oversized_doc() {
        let mut md = String::from("# Doc\n\n## Big\n\n");
        for i in 0..50 {
            md.push_str(&format!(
                "Paragraph number {}. It has some content. Filler text to inflate. ",
                i
            ));
            md.push_str("\n\n");
        }
        md.push_str("## Small\n\nA tiny tail section.\n");
        let chunks = chunk_markdown(&md, ChunkOpts::default());
        assert!(chunks.len() > 1);
        assert_offsets_recoverable(&md, &chunks);
    }

    #[test]
    fn offsets_are_recoverable_flat_blob() {
        let mut md = String::from("## References\n\n");
        for i in 0..2000 {
            md.push_str(&format!("[ref{}]", i));
        }
        md.push('\n');
        let chunks = chunk_markdown(&md, ChunkOpts::default());
        assert!(chunks.len() > 1);
        assert_offsets_recoverable(&md, &chunks);
    }

    #[test]
    fn heading_path_without_h1() {
        // Document opens at H2 (no top-level H1) — common in PDF extraction
        // where the title is in metadata. The H2 should appear at index 0
        // of heading_path, not be padded with an empty string.
        let md = "## Section\n\nbody.\n\n### Subsection\n\nmore body.\n";
        let chunks = chunk_markdown(md, ChunkOpts::default());
        for c in &chunks {
            assert!(
                c.heading_path.iter().all(|s| !s.is_empty()),
                "heading_path has empty entries: {:?}",
                c.heading_path
            );
        }
        // The subsection chunk should have both headings in scope
        let sub = chunks.iter().find(|c| {
            c.heading_path
                .last()
                .is_some_and(|s| s.contains("Subsection"))
        });
        assert!(sub.is_some(), "no subsection chunk found");
        let path = &sub.unwrap().heading_path;
        assert_eq!(path.len(), 2);
        assert!(path[0].contains("Section"));
        assert!(path[1].contains("Subsection"));
    }

    #[test]
    fn flat_unstructured_blob_is_hard_split() {
        // Simulates a citations section: one giant section with no
        // paragraph breaks and no sentence-final patterns.
        let mut md = String::from("## References\n\n");
        for i in 0..3000 {
            md.push_str(&format!("[citation{}]", i));
        }
        md.push('\n');
        let chunks = chunk_markdown(&md, ChunkOpts::default());
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.char_length <= DEFAULT_TARGET_CHARS + DEFAULT_OVERLAP_CHARS,
                "chunk {} too large: {}",
                c.ordinal,
                c.char_length
            );
        }
    }

    #[test]
    fn oversized_section_is_split() {
        let mut md = String::from("## Big\n\n");
        for i in 0..50 {
            md.push_str(&format!("Paragraph number {}. It has some content. ", i));
            md.push_str("Some more filler text to inflate the section. ");
            md.push_str("\n\n");
        }
        let chunks = chunk_markdown(&md, ChunkOpts::default());
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.char_length <= DEFAULT_TARGET_CHARS + DEFAULT_OVERLAP_CHARS * 2,
                "chunk too large: {}",
                c.char_length
            );
        }
    }
}
