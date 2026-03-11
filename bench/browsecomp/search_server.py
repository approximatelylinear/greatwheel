"""Thin HTTP server wrapping BM25sSearcher for gw-bench.

Usage:
    uv run --project bench/browsecomp --extra bm25s python bench/browsecomp/search_server.py \
        --index-path vendor/BrowseComp-Plus/data/bm25s-index --port 8000

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
    searcher: BM25sSearcher = None  # set after init

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/call/search":
            query = body.get("query", "")
            k = body.get("k", 5)
            results = self.searcher.search(query, k=k)
            # Return snippet (first 1000 chars) instead of full text
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
            doc = self.searcher.get_document(docid)
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
    args = parser.parse_args()

    print(f"Loading BM25s index from {args.index_path} ...", flush=True)
    searcher = BM25sSearcher(args)
    SearchHandler.searcher = searcher

    server = HTTPServer((args.host, args.port), SearchHandler)
    print(f"Search server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
