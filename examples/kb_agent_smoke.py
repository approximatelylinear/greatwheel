#!/usr/bin/env python3
"""
Interactive smoke test for gw-kb agent integration.

Requires a running `greatwheel` server (`cargo run --bin greatwheel`)
with `kb.enabled = true` in config/greatwheel.toml and KB data
populated via `gw-kb` CLI. Talks to the session API via HTTP.

Usage:
    python examples/kb_agent_smoke.py

    # Or drive it with your own query:
    python examples/kb_agent_smoke.py "how does PLAID speed up late interaction?"

What it does:
    1. Creates a new session.
    2. Sends a user message with explicit instructions telling the
       LLM to use `kb_search` and/or `kb_explore` from the host API.
    3. Prints the response, the session tree, and any host function
       traces that ouros emits.

This is the "interactive" side of end-to-end validation — the
deterministic Rust integration test (crates/gw-kb/tests/agent_integration.rs)
proves the plumbing works; this script proves an LLM-driven agent can
actually call KB functions from generated code.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SERVER = "http://localhost:8090"

# Default query if none is passed on the command line.
DEFAULT_QUERY = (
    "Search the knowledge base for efficient inference techniques in "
    "large language models and summarize the top two findings."
)

# Instructions wrapped around the user query. The LLM running inside
# greatwheel's conversation loop sees this as the user turn. The
# explicit mention of kb_search / kb_topic tells the model these are
# available external functions.
AGENT_INSTRUCTIONS = """\
You have access to a knowledge base via these host functions:

- kb_search(query: str, k: int = 5) -> list[dict]
    Returns ranked chunks. Each dict has:
      chunk_id, source_id, source_title, source_url, heading_path,
      content, score.

- kb_explore(query: str, k: int = 15) -> list[dict]
    Spreading-activation discovery over the topic graph. Each dict has:
      topic_id, label, slug, chunk_count, score.

- kb_topic(slug: str) -> dict | None
    Fetches one topic with its summary and neighbor graph.

- kb_topics(limit: int = 50) -> list[dict]
    Lists topics.

Use these to answer the user's question. Return your answer via
FINAL("...").

User question:
{query}
"""


def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach {SERVER} — is `greatwheel` running? ({e})"
        ) from None


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    print(f"→ server:   {SERVER}")
    print(f"→ query:    {query}\n")

    print("1. Creating session...")
    create = post("/api/sessions/create", {})
    sid = create["session_id"]
    print(f"   session_id = {sid}\n")

    message = AGENT_INSTRUCTIONS.format(query=query)

    print("2. Sending message (this will run the agent through the LLM loop)...")
    t0 = time.time()
    try:
        result = post(
            "/api/sessions/message",
            {"session_id": sid, "message": message},
        )
    except RuntimeError as e:
        print(f"   FAILED: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0
    print(f"   elapsed = {elapsed:.1f}s\n")

    print("3. Agent response:")
    print("-" * 60)
    print(result.get("response", "(no response)"))
    print("-" * 60)
    print(
        f"   is_final    = {result.get('is_final')}\n"
        f"   iterations  = {result.get('iterations')}\n"
        f"   in_tokens   = {result.get('input_tokens')}\n"
        f"   out_tokens  = {result.get('output_tokens')}"
    )

    print("\n4. Fetching session tree (to inspect what the agent did)...")
    tree = post("/api/sessions/tree", {"session_id": sid})
    entries = tree.get("entries", [])
    print(f"   {len(entries)} entries")

    # Show each entry's type and a short preview of its content. Useful
    # for seeing whether the agent wrote Python that actually called
    # kb_search, what came back, and how it reasoned.
    for i, e in enumerate(entries):
        kind = e.get("entry_type", "?")
        content = e.get("content", "") or ""
        # Trim long entries to keep output scannable.
        if isinstance(content, str) and len(content) > 300:
            content = content[:300] + "…"
        print(f"   [{i}] {kind}: {content!r}")

    print("\n5. Ending session...")
    post("/api/sessions/end", {"session_id": sid})
    print("   done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
