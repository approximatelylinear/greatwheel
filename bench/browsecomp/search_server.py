"""Thin HTTP server wrapping BM25s (+ optional LanceDB vector) search for gw-bench.

Usage:
    # BM25-only (default):
    uv run --project bench/browsecomp --extra bm25s python bench/browsecomp/search_server.py \
        --index-path vendor/BrowseComp-Plus/data/bm25s-index --port 8000

    # Hybrid BM25 + LanceDB vector search:
    uv run --project bench/browsecomp --extra bm25s python bench/browsecomp/search_server.py \
        --index-path vendor/BrowseComp-Plus/data/bm25s-index \
        --lancedb-path vendor/BrowseComp-Plus/data/lancedb \
        --port 8000

Endpoints:
    POST /call/search       {"query": "...", "k": 5}  -> [{"docid": ..., "snippet": ...}, ...]
    POST /call/get_document {"docid": "..."}           -> full document text
"""

import argparse
import json
import logging
import sys
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(__file__))
from bm25s_searcher import BM25sSearcher


def reciprocal_rank_fusion(result_lists, k=60):
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list is a list of dicts with at least "docid" and "text" keys.
    Returns merged list sorted by RRF score (highest first).
    """
    scores = {}  # docid -> RRF score
    docs = {}    # docid -> result dict

    for results in result_lists:
        for rank, r in enumerate(results):
            docid = r["docid"]
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank + 1)
            if docid not in docs:
                docs[docid] = r

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [docs[docid] for docid, _ in ranked]


class SearchHandler(BaseHTTPRequestHandler):
    bm25_searcher: BM25sSearcher = None
    lance_searcher = None  # Optional LanceDBSearcher

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/call/search":
            query = body.get("query", "")
            k = body.get("k", 5)

            # BM25 results
            bm25_results = self.bm25_searcher.search(query, k=k)

            if self.lance_searcher is not None:
                # BM25-first hybrid: return all k BM25 results, then
                # append up to k/2 vector-unique results (documents
                # BM25 missed but vector search found)
                try:
                    vec_results = self.lance_searcher.search(query, k=k)
                except Exception:
                    vec_results = []

                if vec_results:
                    bm25_docids = {r["docid"] for r in bm25_results}
                    vec_extra = [r for r in vec_results if r["docid"] not in bm25_docids]
                    # BM25 results first (full k), then vector extras
                    results = list(bm25_results) + vec_extra[:max(k // 2, 3)]
                else:
                    results = bm25_results
            else:
                results = bm25_results

            out = []
            for r in results:
                out.append({
                    "docid": r["docid"],
                    "score": r.get("score"),
                    "snippet": r.get("text", "")[:3000],
                })
            self._json_response(200, out)

        elif self.path == "/call/get_document":
            docid = body.get("docid", "")
            doc = self.bm25_searcher.get_document(docid)
            if doc is None and self.lance_searcher is not None:
                doc = self.lance_searcher.get_document(docid)
            if doc:
                self._json_response(200, doc["text"])
            else:
                self._json_response(404, {"error": f"docid {docid} not found"})

        elif self.path == "/":
            self._json_response(200, {"status": "ok"})

        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        self._json_response(200, {"status": "ok"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    BM25sSearcher.parse_args(parser)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--lancedb-path",
        default=None,
        help="Path to LanceDB database directory (enables hybrid search)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama URL for embeddings (used with --lancedb-path)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("GW_EMBED_MODEL", "nomic-embed-text"),
        help="Embedding model (used with --lancedb-path)",
    )
    args = parser.parse_args()

    print(f"Loading BM25s index from {args.index_path} ...", flush=True)
    searcher = BM25sSearcher(args)
    SearchHandler.bm25_searcher = searcher

    if args.lancedb_path:
        from lancedb_searcher import LanceDBSearcher
        lance_args = argparse.Namespace(
            db_path=args.lancedb_path,
            table_name="browsecomp_docs",
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
        )
        print(f"Loading LanceDB from {args.lancedb_path} ...", flush=True)
        SearchHandler.lance_searcher = LanceDBSearcher(lance_args)
        mode = "hybrid (BM25 + LanceDB vector)"
    else:
        mode = "BM25-only"

    server = HTTPServer((args.host, args.port), SearchHandler)
    print(f"Search server listening on {args.host}:{args.port} ({mode})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
