"""Extraction helpers for gw-kb.

Each submodule exposes an `extract(...)` function that returns a dict with:
    {
        "markdown": str,
        "title": str | None,
        "author": str | None,
        "published_at": str | None,   # ISO 8601
        "source_format": str,         # "html" | "pdf" | "markdown"
        "extractor": str,             # tool name
    }
"""
