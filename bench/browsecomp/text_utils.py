"""Shared text utilities for BrowseComp corpus processing."""

CHUNK = 4096
OVERLAP = 800


def split(text):
    """Chunk text into overlapping passages of CHUNK chars with OVERLAP overlap."""
    if len(text) <= CHUNK:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - OVERLAP
    return out


def title(text):
    """Extract title from YAML frontmatter."""
    in_fm = False
    for line in text.split("\n"):
        s = line.strip()
        if s == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm and s.startswith("title:"):
            return s.split(":", 1)[1].strip()
    return ""
