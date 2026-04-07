"""Markdown / plaintext passthrough."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def extract(path: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    title = None
    author = None
    published_at = None

    # Strip YAML frontmatter if present
    m = _FRONTMATTER_RE.match(text)
    if m:
        frontmatter, body = m.group(1), m.group(2)
        text = body
        for line in frontmatter.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if key == "title":
                title = value
            elif key == "author":
                author = value
            elif key in ("date", "published", "published_at"):
                published_at = value

    # Fallback title: first H1
    if title is None:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

    # Fallback title: filename
    if title is None:
        title = p.stem

    return {
        "markdown": text,
        "title": title,
        "author": author,
        "published_at": published_at,
        "source_format": "markdown",
        "extractor": "passthrough",
    }
