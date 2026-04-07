"""PDF extraction via pymupdf4llm.

For now we only use pymupdf4llm — fast, no ML dependencies, handles
born-digital PDFs well. Marker can be added later as a fallback for
scanned/complex layouts.
"""

from __future__ import annotations

from typing import Any

import pymupdf4llm


def extract(path: str) -> dict[str, Any]:
    """Extract markdown from a PDF file."""
    markdown = pymupdf4llm.to_markdown(path)
    if not markdown:
        raise RuntimeError(f"pymupdf4llm returned no content for {path}")

    # pymupdf4llm doesn't expose metadata directly, but we can pull it
    # from the underlying pymupdf doc
    import pymupdf

    doc = pymupdf.open(path)
    meta = doc.metadata or {}
    title = meta.get("title") or None
    author = meta.get("author") or None
    # creationDate is in PDF date format like "D:20240315120000+00'00'"
    published_at = _parse_pdf_date(meta.get("creationDate"))
    doc.close()

    return {
        "markdown": markdown,
        "title": title,
        "author": author,
        "published_at": published_at,
        "source_format": "pdf",
        "extractor": "pymupdf4llm",
    }


def _parse_pdf_date(raw: str | None) -> str | None:
    """Convert PDF date format (D:YYYYMMDDHHMMSS...) to ISO 8601."""
    if not raw:
        return None
    s = raw[2:] if raw.startswith("D:") else raw
    if len(s) < 8:
        return None
    try:
        year = s[0:4]
        month = s[4:6]
        day = s[6:8]
        hour = s[8:10] if len(s) >= 10 else "00"
        minute = s[10:12] if len(s) >= 12 else "00"
        second = s[12:14] if len(s) >= 14 else "00"
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
    except Exception:
        return None
