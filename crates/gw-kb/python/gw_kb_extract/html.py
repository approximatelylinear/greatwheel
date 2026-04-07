"""HTML extraction via trafilatura."""

from __future__ import annotations

import json
from typing import Any

import trafilatura
from trafilatura.settings import use_config


_CONFIG = use_config()
_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")


def extract(url: str, html: str | None = None) -> dict[str, Any]:
    """Extract clean markdown + metadata from an HTML page.

    If `html` is None, trafilatura fetches the URL itself.
    """
    if html is None:
        html = trafilatura.fetch_url(url)
        if html is None:
            raise RuntimeError(f"failed to fetch {url}")

    # Extract content as markdown
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        include_formatting=True,
        with_metadata=False,
        config=_CONFIG,
    )

    if not markdown:
        raise RuntimeError(f"trafilatura returned no content for {url}")

    # Extract metadata as JSON
    metadata_json = trafilatura.extract_metadata(html)
    title = None
    author = None
    published_at = None
    if metadata_json is not None:
        meta_dict = metadata_json.as_dict() if hasattr(metadata_json, "as_dict") else {}
        title = meta_dict.get("title")
        author = meta_dict.get("author")
        published_at = meta_dict.get("date")  # already ISO-ish

    return {
        "markdown": markdown,
        "title": title,
        "author": author,
        "published_at": published_at,
        "source_format": "html",
        "extractor": "trafilatura",
    }
