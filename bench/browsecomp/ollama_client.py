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

# Local imports
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
from fact_registry import FactRegistry
LOCAL_SEARCHERS = {"bm25s", "lancedb-local", "gw-kb"}  # names we handle ourselves


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

SYSTEM_PROMPT = """You are a research agent answering questions by searching a corpus of ~100K web documents using a REPL environment.

Your REPL has these built-in functions:
- search(query) -> list of {docid, score, snippet} dicts. BM25 keyword search — use 2-4 specific terms.
- get_document(docid) -> full document text as string
- llm_query(prompt) -> ask a sub-LLM to analyze text (useful for long documents)
- entity_search(entity, hops=1, k=5) -> one-hop entity chain search. Use for multi-hop questions.

STRATEGY:
1. Identify the most distinctive clues (specific dates, places, events, names)
2. Search for those clues with SHORT keyword queries (2-4 terms)
3. Read promising documents with get_document()
4. Use llm_query() to analyze long documents: llm_query(f"Who is mentioned? {doc[:5000]}")
5. Use discovered names/facts to search again
6. Chain your findings until you can answer
7. SUBMIT YOUR ANSWER as soon as you have a plausible candidate

Example:
```python
results = search("convent Michigan 1932")
for r in results:
    print(r['docid'], r['snippet'][:200])
```
```python
doc = get_document("12345")
answer = llm_query(f"What person is described in this article? {doc[:5000]}")
print(answer)
```
```python
# Search for discovered entity
results = search("Jane Smith Michigan convent")
for r in results:
    print(r['docid'], r['snippet'][:200])
```

RULES:
- NEVER answer "Unable to determine". Always give your BEST GUESS from the documents.
- NEVER repeat a similar search. Each search must use different keywords.
- ALWAYS read at least 2 full documents before submitting.
- A wrong answer is better than no answer. Submit EARLY.

When you have enough evidence, provide your answer as:
Exact Answer: <precise answer>"""

QUERY_TEMPLATE = """Question: {question}

Search the corpus step by step. Use short keyword queries. Read full documents. Use llm_query() to analyze long text.
When done: Exact Answer: <answer>"""

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
    """Agent that uses Ollama or SGLang for reasoning and a BrowseComp-Plus searcher for retrieval."""

    def __init__(
        self,
        ollama_url: str,
        model: str,
        searcher: Any,
        k: int = 5,
        max_turns: int = 10,
        include_get_document: bool = False,
        snippet_max_chars: int = 2000,
        backend: str = "ollama",
        system_prompt: str | None = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.searcher = searcher
        self.k = k
        self.max_turns = max_turns
        self.include_get_document = include_get_document
        self.snippet_max_chars = snippet_max_chars
        self.backend = backend
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request and return a normalized response.

        Returns a dict with:
          - message.content (str)
          - prompt_eval_count (int) — input tokens
          - eval_count (int) — output tokens
        Regardless of backend, the response shape is normalized to this format.
        """
        if self.backend == "sglang":
            return self._chat_openai(messages, tools)
        return self._chat_ollama(messages, tools)

    def _chat_ollama(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request via Ollama's native /api/chat endpoint."""
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

    def _chat_openai(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat request via OpenAI-compatible /v1/chat/completions (SGLang/vLLM)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.ollama_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        # Normalize to Ollama-style response shape
        usage = data.get("usage", {})
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
        return {
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    def _execute_tool(self, name: str, arguments: dict, repl_namespace: dict | None = None, context: dict | None = None, tool_counts: dict | None = None) -> str:
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
            return self._execute_python(code, repl_namespace or {})

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _build_repl_namespace(self, context: dict, tool_counts: dict) -> dict:
        """Build the persistent REPL namespace with search/get_document/llm_query functions."""

        def search(query: str, k: int | None = None) -> list[dict]:
            """Search the corpus. BM25 keyword matching — use 2-4 specific terms."""
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

        def llm_query(prompt: str) -> str:
            """Query a sub-LLM to analyze text. Useful for extracting info from long documents."""
            try:
                messages = [{"role": "user", "content": prompt[:8000]}]
                if self.backend == "sglang":
                    resp = requests.post(
                        f"{self.ollama_url}/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("choices"):
                        return data["choices"][0].get("message", {}).get("content", "")
                    return ""
                else:
                    resp = requests.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    return resp.json().get("message", {}).get("content", "")
            except Exception as e:
                return f"Error: {e}"

        ns = {
            "search": search,
            "get_document": get_document,
            "llm_query": llm_query,
            "facts": FactRegistry(),
            "search_results": context["search_results"],
            "documents": context["documents"],
            "re": re,
            "json": json,
        }
        # Bootstrap entity_search() — needs search() and _extract_entities() in namespace
        entity_search_path = BENCH_DIR / "entity_search.py"
        with open(entity_search_path, encoding="utf-8") as f:
            exec(f.read(), ns)  # noqa: S102

        # Searchers can inject extra REPL functions (e.g. gw-kb's
        # kb_topic/kb_topics/kb_explore). Each call gets wrapped in a
        # counter so the per-query JSON's tool_call_counts attribute lift
        # to specific tools, not just "more tools".
        if hasattr(self.searcher, "repl_extras"):
            for extra_name, extra_fn in self.searcher.repl_extras().items():
                def _wrap(fn=extra_fn, n=extra_name):
                    def wrapped(*args, **kwargs):
                        tool_counts[n] = tool_counts.get(n, 0) + 1
                        return fn(*args, **kwargs)
                    return wrapped
                ns[extra_name] = _wrap()
        return ns

    def _execute_python(self, code: str, namespace: dict) -> str:
        """Execute Python code in the persistent REPL namespace."""
        import io
        import contextlib

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

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract code from ```python or ```repl fenced blocks."""
        blocks = []
        for m in re.finditer(r'```(?:python|repl|py)?\s*\n(.*?)```', text, re.DOTALL):
            code = m.group(1).strip()
            if code:
                blocks.append(code)
        return blocks

    def run(self, query: str, query_id: str | None = None) -> dict:
        """Run the full agent loop using text-based REPL (no tool calling)."""
        formatted_query = QUERY_TEMPLATE.format(question=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_query},
        ]

        results: list[dict] = []
        tool_counts: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        status = "completed"

        # Persistent REPL namespace
        python_context: dict = {"search_results": [], "documents": {}}
        repl_namespace = self._build_repl_namespace(python_context, tool_counts)

        for turn in range(self.max_turns):
            try:
                # No tools — just text generation
                response = self._chat(messages)
            except Exception as e:
                print(f"  [Error] Ollama call failed on turn {turn}: {e}")
                status = "error"
                break

            msg = response.get("message", {})
            content = msg.get("content", "")

            # Track tokens
            total_input_tokens += response.get("prompt_eval_count", 0)
            total_output_tokens += response.get("eval_count", 0)

            if content:
                results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": content,
                })

            # Check for "Exact Answer:" anywhere in the response (including
            # after code blocks, in comments, or in plain text)
            if "exact answer:" in content.lower():
                break

            # Extract and execute code blocks from the response
            code_blocks = self._extract_code_blocks(content)

            if not code_blocks:
                # Steer the model
                if turn == 0:
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Write Python code in a ```python block to search the corpus. "
                            "Use search('keyword1 keyword2') with 2-4 terms from the question."
                        ),
                    })
                    continue
                # Ask for final answer
                messages.append(msg)
                messages.append({
                    "role": "user",
                    "content": "Based on what you've found, provide your final answer as: Exact Answer: <answer>",
                })
                continue

            # Execute each code block and collect output
            messages.append(msg)
            all_outputs = []
            for code in code_blocks:
                output = self._execute_python(code, repl_namespace)
                results.append({
                    "type": "tool_call",
                    "tool_name": "python",
                    "arguments": {"code": code},
                    "output": output,
                })
                tool_counts["python"] = tool_counts.get("python", 0) + 1
                all_outputs.append(output)

            # Feed REPL output back as a user message with turn-aware nudging
            repl_output = "\n".join(f"[REPL output]\n{o}" for o in all_outputs)
            if len(repl_output) > 6000:
                repl_output = repl_output[:6000] + "\n... (truncated)"

            if turn >= self.max_turns - 2:
                nudge = "\n\nLAST CHANCE — you MUST provide your answer NOW as: Exact Answer: <answer>"
            elif turn >= self.max_turns // 2:
                nudge = "\n\nYou should have a candidate by now. Provide your answer as: Exact Answer: <answer>"
            else:
                nudge = "\n\nContinue your analysis. Write more code or provide your answer as: Exact Answer: <answer>"

            messages.append({
                "role": "user",
                "content": repl_output + nudge,
            })
        else:
            status = "max_turns"
            # Fallback: try to extract answer from facts.best_candidate() or
            # ask the model one more time with a forced answer prompt
            best = repl_namespace.get("facts", None)
            if best is not None:
                try:
                    candidate = best.best_candidate()
                    if candidate:
                        results.append({
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": f"Exact Answer: {candidate[0]}",
                        })
                        status = "fallback_facts"
                except Exception:
                    pass

        record = {
            "metadata": {
                "model": self.model,
                "backend": self.backend,
                "llm_url": self.ollama_url,
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
    elif name == "gw-kb":
        from gw_kb_searcher import GwKbSearcher
        return GwKbSearcher
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
        description="Greatwheel BrowseComp-Plus agent client (Ollama/SGLang)"
    )

    # Agent config
    parser.add_argument(
        "--backend",
        choices=["ollama", "sglang"],
        default=os.getenv("GW_LLM_BACKEND", "ollama"),
        help="LLM backend: ollama (native API) or sglang (OpenAI-compatible). "
             "(default: $GW_LLM_BACKEND or ollama)",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        help="LLM server URL. Overrides --ollama-url. "
             "(default: $GW_LLM_URL or backend-specific default)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API URL (default: $OLLAMA_URL or http://localhost:11434)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GW_MODEL", "qwen2.5:7b"),
        help="Model name (default: $GW_MODEL or qwen2.5:7b)",
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
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a file containing the system prompt. Overrides the default.",
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

    # Resolve LLM URL: --llm-url > $GW_LLM_URL > --ollama-url (with backend-specific default)
    llm_url = args.llm_url or os.getenv("GW_LLM_URL")
    if not llm_url:
        if args.backend == "sglang":
            llm_url = os.getenv("SGLANG_URL", "http://localhost:30000")
        else:
            llm_url = args.ollama_url

    # Load system prompt from file if requested
    system_prompt: str | None = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"Loaded system prompt from {args.system_prompt_file} ({len(system_prompt)} chars)")

    # Initialize agent
    agent = OllamaAgent(
        ollama_url=llm_url,
        model=args.model,
        searcher=searcher,
        k=args.k,
        max_turns=args.max_turns,
        include_get_document=args.include_get_document,
        snippet_max_chars=args.snippet_max_chars,
        backend=args.backend,
        system_prompt=system_prompt,
    )
    print(f"Agent ready (backend={args.backend}, model={args.model}, url={llm_url}, max_turns={args.max_turns}, k={args.k})")

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
