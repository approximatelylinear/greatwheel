"""Blob-store-backed ColBERT rerank server — drop-in for rerank_server.py.

Same HTTP API as rerank_server.py (POST /rerank) but uses precomputed
token tensors from the passage blob store instead of re-encoding on GPU.
~200ms/query vs ~90s/query.

This is the correct way to test "BM25 + ColBERT rerank" end-to-end with
the native Rust backend:
  1. gw-bench uses --search-backend native (tantivy with boosts)
  2. gw-bench uses --rerank-url http://localhost:8001 (this server)
  3. This server fetches precomputed tensors from the blob store and
     computes MaxSim in ~200ms

Usage:
    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/rerank_server_blobs.py \\
        --blob-store data/passage-blobs --port 8001

    cargo run --bin gw-bench -- \\
        --search-backend native \\
        --tantivy-index vendor/BrowseComp-Plus/data/tantivy-corpus \\
        --rerank-url http://localhost:8001 \\
        --model qwen3.5:9b ...
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(__file__))
from blob_reranker import BlobReranker


class BlobRerankHandler(BaseHTTPRequestHandler):
    reranker: BlobReranker = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/rerank":
            query = body.get("query", "")
            documents = body.get("documents", [])
            k = body.get("k", 10)

            if not query or not documents:
                self._json_response(200, [])
                return

            # The native backend sends {"docid": "...", "text": "..."} per doc.
            # BlobReranker.rerank() expects [{"docid": "..."}] and looks up
            # tokens from the blob store internally.
            candidates = [{"docid": d["docid"]} for d in documents]
            reranked = self.reranker.rerank(query, candidates)

            # Return in the same format the native backend expects:
            # [{"docid": "...", "score": float, "text": "..."}, ...]
            # Preserve the original text from the BM25 candidates.
            text_by_docid = {d["docid"]: d.get("text", "") for d in documents}
            out = []
            for r in reranked[:k]:
                out.append({
                    "docid": r["docid"],
                    "score": r["score"],
                    "text": text_by_docid.get(r["docid"], ""),
                })
            self._json_response(200, out)

        elif self.path == "/":
            self._json_response(200, {"status": "ok", "backend": "blob_rerank"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        self._json_response(200, {"status": "ok", "backend": "blob_rerank"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Blob-store ColBERT rerank server")
    parser.add_argument("--blob-store", default="data/passage-blobs",
                        help="Path to passage blob store (Lance dir)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Loading blob reranker from {args.blob_store} ...", flush=True)
    BlobRerankHandler.reranker = BlobReranker(args.blob_store)

    server = HTTPServer((args.host, args.port), BlobRerankHandler)
    print(f"Blob rerank server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
