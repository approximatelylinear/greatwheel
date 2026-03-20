"""Lightweight HTTP server that ONLY does ColBERT reranking.

BM25 retrieval happens in Rust (native tantivy). This server receives
pre-retrieved candidates and reranks them using ColBERT MaxSim scoring.

Usage:
    uv run --project bench/browsecomp --extra colbert python bench/browsecomp/rerank_server.py \
        --model lightonai/Reason-ModernColBERT --port 8001

Endpoint:
    POST /rerank  {"query": "...", "documents": [{"docid": "...", "text": "..."}, ...], "k": 10}
                  -> [{"docid": "...", "score": 0.123, "text": "..."}, ...]
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from http.server import HTTPServer, BaseHTTPRequestHandler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from colbert_reranker import ColBERTReranker


class RerankHandler(BaseHTTPRequestHandler):
    reranker: ColBERTReranker = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/rerank":
            query = body.get("query", "")
            documents = body.get("documents", [])
            k = body.get("k", 10)

            results = self.reranker.rerank(query, documents, k=k)

            out = []
            for r in results:
                out.append({
                    "docid": r["docid"],
                    "score": r.get("score"),
                    "text": r.get("text", ""),
                })
            self._json_response(200, out)

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
    parser = argparse.ArgumentParser(description="ColBERT rerank-only server")
    parser.add_argument("--model", default="lightonai/Reason-ModernColBERT",
                        help="ColBERT model name")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Loading ColBERT model: {args.model} ...", flush=True)
    RerankHandler.reranker = ColBERTReranker(args.model)

    server = HTTPServer((args.host, args.port), RerankHandler)
    print(f"Rerank server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
