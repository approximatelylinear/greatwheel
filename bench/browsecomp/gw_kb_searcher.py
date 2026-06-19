"""Searcher backed by `gw-kb serve` over localhost HTTP.

Plugs into `ollama_client.py` like the other local searchers (bm25s,
lancedb-local). The standard `search` / `get_document` interface is
implemented as required by the agent loop, and `repl_extras()` exposes
gw-kb's topic-aware tools (`kb_topic`, `kb_topics`, `kb_explore`) under
ablation gating so we can attribute lift to those features specifically.

`get_document` reads from an in-memory dict loaded from the BrowseComp
jsonl at startup. gw-kb stores chunks not whole documents, so SQL
reconstruction would either lose original ordering or duplicate
overlap-chars. The sample30 corpus is small enough that loading the
full jsonl into a dict is the simpler, lossless option.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests


class GwKbSearcher:
    """Calls `gw-kb serve` and exposes topic-aware tools to the agent."""

    search_type = "gw-kb"

    def __init__(self, args: Any):
        self.base_url: str = args.gw_kb_url.rstrip("/")
        self.tools_mode: str = args.gw_kb_tools
        self.session = requests.Session()
        self.session.headers["content-type"] = "application/json"

        self.docs = self._load_docs(args.bc_jsonl)
        print(f"Loaded {len(self.docs)} unique BrowseComp docs from {args.bc_jsonl}")

        self._wait_for_health()

    # ── Loaders ───────────────────────────────────────────────────────

    def _load_docs(self, path: str) -> dict[str, dict]:
        """Read the BrowseComp jsonl into a {docid: {text, url, title}} dict.

        Each line has the shape:
            {query_id, query, answer, gold_docs, negative_docs, evidence_docs}
        with each `*_docs` entry being `{docid, text, url}`. We dedup
        across queries by docid (negatives appear many times)."""
        docs: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("gold_docs", "negative_docs", "evidence_docs"):
                    for d in obj.get(key, []) or []:
                        did = str(d.get("docid", ""))
                        if not did or did in docs:
                            continue
                        docs[did] = {
                            "text": d.get("text", ""),
                            "url": d.get("url", ""),
                            "title": d.get("title", ""),
                        }
        return docs

    def _wait_for_health(self, timeout: float = 30.0) -> None:
        """Poll /healthz until the server answers or we give up."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self.session.get(f"{self.base_url}/healthz", timeout=2)
                if r.ok:
                    body = r.json()
                    print(
                        f"gw-kb healthz: ok=true topics={body.get('topics')} "
                        f"sources={body.get('sources')}"
                    )
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(
            f"gw-kb serve at {self.base_url} did not become healthy within {timeout}s"
        )

    # ── Standard searcher interface ───────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[dict]:
        r = self.session.post(
            f"{self.base_url}/search",
            json={"query": query, "k": k},
            timeout=60,
        )
        r.raise_for_status()
        out = []
        for hit in r.json():
            out.append(
                {
                    "docid": hit.get("docid") or "",
                    "score": hit.get("score", 0.0),
                    "text": hit.get("content", ""),
                }
            )
        return out

    def get_document(self, docid: str) -> dict | None:
        doc = self.docs.get(str(docid))
        if doc is None:
            return None
        return {"docid": docid, "text": doc["text"]}

    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--gw-kb-url",
            default="http://localhost:9099",
            help="gw-kb serve base URL (default: http://localhost:9099)",
        )
        parser.add_argument(
            "--bc-jsonl",
            default="vendor/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
            help="Path to BrowseComp-Plus decrypted jsonl (used by get_document)",
        )
        parser.add_argument(
            "--gw-kb-tools",
            choices=["search-only", "+topic", "+explore", "full"],
            default="full",
            help="Which gw-kb tools the agent gets access to (ablation control)",
        )

    # ── Topic-aware extras (ablation-gated) ───────────────────────────

    def repl_extras(self) -> dict:
        extras: dict = {}
        if self.tools_mode in ("+topic", "+explore", "full"):
            extras["kb_topic"] = self._kb_topic
            extras["kb_topics"] = self._kb_topics
        if self.tools_mode in ("+explore", "full"):
            extras["kb_explore"] = self._kb_explore
        return extras

    def _kb_topic(self, slug: str) -> dict | None:
        r = self.session.post(
            f"{self.base_url}/topic", json={"slug": slug}, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def _kb_topics(self, limit: int = 50) -> list[dict]:
        r = self.session.post(
            f"{self.base_url}/topics", json={"limit": limit}, timeout=30
        )
        r.raise_for_status()
        return r.json()

    def _kb_explore(self, query: str, k: int = 15) -> list[dict]:
        r = self.session.post(
            f"{self.base_url}/explore",
            json={"query": query, "k": k, "seeds": 3, "hops": 3, "decay": 0.5},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
