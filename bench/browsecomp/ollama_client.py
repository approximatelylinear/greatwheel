#!/usr/bin/env python3
"""
Greatwheel BrowseComp-Plus agent client using Ollama.

Follows the same output format as the BrowseComp-Plus search_agent clients
so results can be evaluated with their evaluate_run.py script.

Usage:
    # Single query
    python ollama_client.py --query "Who won the 2024 Nobel Prize in Physics?" \
        --searcher-type bm25 --index-path /path/to/lucene-index

    # Full dataset
    python ollama_client.py --query topics-qrels/queries.tsv \
        --searcher-type bm25 --index-path /path/to/lucene-index
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# Add the BrowseComp-Plus repo to the path so we can import their searcher
VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "BrowseComp-Plus"
sys.path.insert(0, str(VENDOR_ROOT / "searcher"))
sys.path.insert(0, str(VENDOR_ROOT / "search_agent"))

from utils import extract_retrieved_docids_from_result

# Local searcher implementations (not in BrowseComp-Plus)
BENCH_DIR = Path(__file__).resolve().parent
LOCAL_SEARCHERS = {"bm25s", "lancedb-local"}  # names we handle ourselves


def _get_vendor_searcher_choices() -> list[str]:
    """Get BrowseComp-Plus searcher type names without triggering heavy imports."""
    # These are the choices defined in SearcherType enum
    return ["bm25", "faiss", "reasonir", "custom"]


def _get_vendor_searcher_class(name: str):
    """Import a BrowseComp-Plus searcher class by name (triggers heavy imports)."""
    from searchers import SearcherType
    return SearcherType.get_searcher_class(name)

# --------------------------------------------------------------------------- #
# Prompt templates (same as BrowseComp-Plus search_agent/prompts.py)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a research agent. You answer questions by writing Python code that searches a corpus "
    "of ~100K web documents.\n\n"
    "YOUR ONLY TOOL is `python`. Inside your code you can call:\n"
    "  search(query) -> list of dicts: [{{'docid': str, 'score': float, 'snippet': str}}, ...]\n"
    "  get_document(docid) -> full document text as string\n\n"
    "CRITICAL RULES FOR BM25 SEARCH:\n"
    "- BM25 matches KEYWORDS, not meaning. Use 2-4 specific terms per query.\n"
    "- NEVER paste the full question as a query. Extract key nouns/names/dates.\n"
    "- Run MULTIPLE searches with DIFFERENT keyword combinations.\n\n"
    "HOW TO SOLVE MULTI-HOP QUESTIONS:\n"
    "These questions describe a chain of facts. You must identify each fact, search for it, "
    "then use what you find to search for the next fact.\n\n"
    "Example approach for: 'What year was the university founded where Person A got their PhD?'\n"
    "```python\n"
    "# Step 1: Find Person A's PhD\n"
    "for r in search('Person A PhD university'):\n"
    "    print(r['docid'], r['snippet'][:200])\n"
    "```\n"
    "Then in the next code block:\n"
    "```python\n"
    "# Step 2: Found that Person A got PhD at MIT. Now find founding year.\n"
    "for r in search('MIT founded year'):\n"
    "    print(r['docid'], r['snippet'][:200])\n"
    "```\n\n"
    "IMPORTANT:\n"
    "- Use print() so you can see results and reason about them.\n"
    "- Read full documents with get_document(docid) when snippets aren't enough.\n"
    "- Use re (regex) to extract specific facts from document text.\n"
    "- Each python call should focus on ONE step of your reasoning chain.\n\n"
    "When you have the answer, respond with ONLY:\n"
    "Exact Answer: <precise answer — name, number, date, or short phrase>"
)

QUERY_TEMPLATE = """Question: {question}

Break this into sub-facts, then search for each one step by step using the python tool.
Use short keyword queries (2-4 terms each). Your final answer must be:
Exact Answer: <answer>"""

# --------------------------------------------------------------------------- #
# Ollama tool-calling interface
# --------------------------------------------------------------------------- #

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the document corpus. Returns top-k results with docid, score, "
            "and snippet. Use different query formulations to find relevant documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

GET_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_document",
        "description": "Retrieve the full text of a document by its docid.",
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {
                    "type": "string",
                    "description": "The document ID to retrieve.",
                }
            },
            "required": ["docid"],
        },
    },
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code. You can call search(query) and get_document(docid) directly in the code. "
            "search(query) returns a list of dicts with keys: docid, score, snippet. "
            "get_document(docid) returns the full document text as a string. "
            "Use print() to output results. You can run multiple searches, filter results, "
            "extract patterns with regex, parse data, and combine information from multiple documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use search() and get_document() to access the corpus.",
                }
            },
            "required": ["code"],
        },
    },
}


class OllamaAgent:
    """Agent that uses Ollama for reasoning and a BrowseComp-Plus searcher for retrieval."""

    def __init__(
        self,
        ollama_url: str,
        model: str,
        searcher: Any,
        k: int = 5,
        max_turns: int = 10,
        include_get_document: bool = False,
        snippet_max_chars: int = 2000,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.searcher = searcher
        self.k = k
        self.max_turns = max_turns
        self.include_get_document = include_get_document
        self.snippet_max_chars = snippet_max_chars

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request to Ollama and return the response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def _execute_tool(self, name: str, arguments: dict, context: dict | None = None, tool_counts: dict | None = None) -> str:
        """Execute a tool call against the searcher and return JSON string result."""
        if name == "search":
            query = arguments.get("query", "")
            results = self.searcher.search(query, self.k)
            # Truncate snippets
            for r in results:
                text = r.get("text", "")
                r["snippet"] = text[: self.snippet_max_chars]
                if "text" in r:
                    del r["text"]
            return json.dumps(results, ensure_ascii=False)

        elif name == "get_document":
            docid = arguments.get("docid", "")
            doc = self.searcher.get_document(str(docid))
            if doc is None:
                return json.dumps({"error": f"Document {docid} not found"})
            return json.dumps(doc, ensure_ascii=False)

        elif name == "python":
            code = arguments.get("code", "")
            return self._execute_python(code, context or {}, tool_counts=tool_counts)

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _execute_python(self, code: str, context: dict, tool_counts: dict | None = None) -> str:
        """Execute Python code with access to search/get_document functions and accumulated context."""
        import io
        import contextlib

        def search(query: str, k: int | None = None) -> list[dict]:
            """Search the corpus from Python code."""
            if tool_counts is not None:
                tool_counts["search"] = tool_counts.get("search", 0) + 1
            results = self.searcher.search(query, k or self.k)
            truncated = []
            for r in results:
                text = r.get("text", "")
                entry = {
                    "docid": r.get("docid", ""),
                    "score": r.get("score", 0.0),
                    "snippet": text[: self.snippet_max_chars],
                }
                truncated.append(entry)
                context["search_results"].append(entry)
            return truncated

        def get_document(docid: str) -> str | None:
            """Retrieve full document text by docid."""
            doc = self.searcher.get_document(str(docid))
            if doc is None:
                return None
            text = doc.get("text", "")
            context["documents"][docid] = text
            return text

        namespace = {
            "search": search,
            "get_document": get_document,
            "search_results": context.get("search_results", []),
            "documents": context.get("documents", {}),
            "re": re,
            "json": json,
        }

        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace)  # noqa: S102
            output = stdout.getvalue()
            if not output:
                output = "(no output)"
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def run(self, query: str, query_id: str | None = None) -> dict:
        """Run the full agent loop for a single query. Returns BrowseComp-Plus format result."""
        tools = [PYTHON_TOOL]

        formatted_query = QUERY_TEMPLATE.format(question=query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_query},
        ]

        results: list[dict] = []
        tool_counts: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        status = "completed"

        # Context for python tool — accumulate search results and documents
        python_context: dict = {"search_results": [], "documents": {}}

        for turn in range(self.max_turns):
            try:
                response = self._chat(messages, tools=tools)
            except Exception as e:
                print(f"  [Error] Ollama call failed on turn {turn}: {e}")
                status = "error"
                break

            msg = response.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Track tokens
            total_input_tokens += response.get("prompt_eval_count", 0)
            total_output_tokens += response.get("eval_count", 0)

            # If the model produced text, record it
            if content:
                results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": content,
                })

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Add assistant message to conversation
            messages.append(msg)

            # Process each tool call
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                arguments = func.get("arguments", {})

                # Execute the tool
                tool_output = self._execute_tool(tool_name, arguments, context=python_context, tool_counts=tool_counts)

                # Accumulate context for python tool
                if tool_name == "search":
                    try:
                        search_hits = json.loads(tool_output)
                        if isinstance(search_hits, list):
                            python_context["search_results"].extend(search_hits)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif tool_name == "get_document":
                    try:
                        doc = json.loads(tool_output)
                        if isinstance(doc, dict) and "docid" in doc:
                            python_context["documents"][doc["docid"]] = doc.get("text", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Record the tool call
                results.append({
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "output": tool_output,
                })
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                # Feed result back to Ollama
                messages.append({
                    "role": "tool",
                    "content": tool_output,
                })
        else:
            # Hit max turns without a final text response
            status = "max_turns"

        record = {
            "metadata": {
                "model": self.model,
                "ollama_url": self.ollama_url,
                "searcher_type": self.searcher.search_type,
                "max_turns": self.max_turns,
                "k": self.k,
            },
            "query_id": query_id,
            "tool_call_counts": tool_counts,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "status": status,
            "retrieved_docids": extract_retrieved_docids_from_result(results),
            "result": results,
        }

        return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _get_local_searcher_class(name: str):
    """Import a local searcher class by name."""
    sys.path.insert(0, str(BENCH_DIR))
    if name == "bm25s":
        from bm25s_searcher import BM25sSearcher
        return BM25sSearcher
    elif name == "lancedb-local":
        from lancedb_searcher import LanceDBSearcher
        return LanceDBSearcher
    else:
        raise ValueError(f"Unknown local searcher: {name}")


def process_single(agent: OllamaAgent, query: str, query_id: str | None, output_dir: Path):
    """Run a single query and save the result."""
    record = agent.run(query, query_id=query_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    qid_part = f"_{query_id}" if query_id else ""
    filename = output_dir / f"run{qid_part}_{ts}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    tool_counts = record.get("tool_call_counts", {})
    print(f"  Saved {filename} | tools: {tool_counts} | status: {record['status']}")


def process_tsv(agent: OllamaAgent, tsv_path: Path, output_dir: Path):
    """Process a TSV file of queries (query_id\tquery)."""
    queries: list[tuple[str, str]] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))

    # Check for already-processed queries
    processed_ids: set[str] = set()
    if output_dir.exists():
        for json_path in output_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid = meta.get("query_id")
                    if qid:
                        processed_ids.add(str(qid))
            except Exception:
                continue

    remaining = [(qid, q) for qid, q in queries if qid not in processed_ids]
    print(f"Processing {len(remaining)} queries (skipping {len(processed_ids)} already done)")

    for qid, query_text in tqdm(remaining, desc="Queries", unit="q"):
        process_single(agent, query_text, qid, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Greatwheel BrowseComp-Plus agent client (Ollama)"
    )

    # Agent config
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API URL (default: $OLLAMA_URL or http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GW_MODEL", "qwen2.5:7b"),
        help="Ollama model name (default: $GW_MODEL or qwen2.5:7b)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agent turns (search-reason cycles) per query (default: 10)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of search results per query (default: 5)",
    )
    parser.add_argument(
        "--include-get-document",
        action="store_true",
        help="Also provide the get_document tool to the agent",
    )
    parser.add_argument(
        "--snippet-max-chars",
        type=int,
        default=2000,
        help="Max characters per search result snippet (default: 2000)",
    )

    # Query input
    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="A query string, or path to a TSV file (query_id\\tquery per line)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="runs/ollama",
        help="Directory to save run result JSON files (default: runs/ollama)",
    )

    # Searcher config — supports both BrowseComp-Plus searchers and our local ones
    all_choices = _get_vendor_searcher_choices() + sorted(LOCAL_SEARCHERS)
    parser.add_argument(
        "--searcher-type",
        choices=all_choices,
        required=True,
        help=f"Searcher backend: {', '.join(all_choices)}",
    )

    # Parse known args first to get searcher type, then add searcher-specific args
    temp_args, _ = parser.parse_known_args()

    if temp_args.searcher_type in LOCAL_SEARCHERS:
        searcher_class = _get_local_searcher_class(temp_args.searcher_type)
    else:
        searcher_class = _get_vendor_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    # Initialize searcher
    print(f"Initializing {args.searcher_type} searcher...")
    searcher = searcher_class(args)
    print(f"Searcher ready ({searcher.search_type})")

    # Initialize agent
    agent = OllamaAgent(
        ollama_url=args.ollama_url,
        model=args.model,
        searcher=searcher,
        k=args.k,
        max_turns=args.max_turns,
        include_get_document=args.include_get_document,
        snippet_max_chars=args.snippet_max_chars,
    )
    print(f"Agent ready (model={args.model}, max_turns={args.max_turns}, k={args.k})")

    # Resolve output dir relative to BrowseComp-Plus vendor dir for evaluation compat
    output_dir = Path(args.output_dir)

    # Check if query is a TSV file
    query_str = args.query.strip()
    potential_path = Path(query_str)
    if not potential_path.is_absolute():
        # Try relative to BrowseComp-Plus vendor dir
        vendor_path = VENDOR_ROOT / query_str
        if vendor_path.is_file():
            potential_path = vendor_path

    if potential_path.is_file() and str(potential_path).endswith(".tsv"):
        process_tsv(agent, potential_path, output_dir)
    else:
        process_single(agent, query_str, None, output_dir)


if __name__ == "__main__":
    main()
