"""Thin HTTP server wrapping BM25s (+ optional reranking) search for gw-bench.

Usage:
    # BM25-only (default):
    uv run --project bench/browsecomp --extra bm25s python bench/browsecomp/search_server.py \
        --index-path vendor/BrowseComp-Plus/data/bm25s-index --port 8000

    # BM25 + blob rerank:
    uv run --project bench/browsecomp --extra bm25s python bench/browsecomp/search_server.py \
        --index-path vendor/BrowseComp-Plus/data/bm25s-index \
        --blob-store vendor/BrowseComp-Plus/data/passage-blobs \
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


class SearchHandler(BaseHTTPRequestHandler):
    bm25_searcher: BM25sSearcher = None
    colbert_reranker = None
    blob_reranker = None

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/call/search":
            query = body.get("query", "")
            k = body.get("k", 5)
            mode = body.get("mode", "bm25")

            if mode == "blob_rerank" and self.blob_reranker is not None:
                bm25_candidates = self.bm25_searcher.search(query, k=200)
                text_by_docid = {c["docid"]: c.get("text", "") for c in bm25_candidates}
                candidate_dicts = [{"docid": c["docid"]} for c in bm25_candidates]
                reranked = self.blob_reranker.rerank(query, candidate_dicts)
                for r in reranked:
                    r["text"] = text_by_docid.get(r["docid"], "")
                results = reranked[:k]
            elif mode == "rerank" and self.colbert_reranker is not None:
                bm25_candidates = self.bm25_searcher.search(query, k=50)
                results = self.colbert_reranker.rerank(query, bm25_candidates, k=k)
            else:
                results = self.bm25_searcher.search(query, k=k)

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
        "--colbert-model",
        default=None,
        help="ColBERT model for reranking (e.g. lightonai/Reason-ModernColBERT-v1)",
    )
    parser.add_argument(
        "--blob-store",
        default=None,
        help="Path to passage blob store (Lance dir). Enables 'blob_rerank' mode: "
             "BM25 top-200 → precomputed ColBERT MaxSim rerank → top-k. "
             "Much faster than --colbert-model (200ms vs 90s) because token "
             "tensors are precomputed.",
    )
    args = parser.parse_args()

    print(f"Loading BM25s index from {args.index_path} ...", flush=True)
    searcher = BM25sSearcher(args)
    SearchHandler.bm25_searcher = searcher

    if args.blob_store:
        from blob_reranker import BlobReranker
        print(f"Loading blob reranker from {args.blob_store} ...", flush=True)
        SearchHandler.blob_reranker = BlobReranker(args.blob_store)
        mode = "BM25 + blob rerank (precomputed ColBERT MaxSim)"
    elif args.colbert_model:
        from colbert_reranker import ColBERTReranker
        print(f"Loading ColBERT reranker: {args.colbert_model} ...", flush=True)
        SearchHandler.colbert_reranker = ColBERTReranker(args.colbert_model)
        mode = f"BM25 + ColBERT rerank ({args.colbert_model})"
    else:
        mode = "BM25-only"

    server = HTTPServer((args.host, args.port), SearchHandler)
    print(f"Search server listening on {args.host}:{args.port} ({mode})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
