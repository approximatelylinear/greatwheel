"""Search server wrapping Qdrant native multi-vector for the BrowseComp agent.

Drop-in replacement for search_server.py — same HTTP API contract
(POST /call/search, POST /call/get_document) so the agent loop doesn't
need any changes.

Requires:
  1. Qdrant running with the colbert_mv collection built
  2. The ColBERT encoder service running (colbert_server.py)
  3. The BM25s index (for get_document text lookups; Qdrant stores vectors
     but not the full doc text)

Usage:
    # Start dependencies:
    docker compose -f docker/docker-compose.bench.yml up -d qdrant
    python bench/browsecomp/colbert_server.py --port 8002 &

    # Start this server:
    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/search_server_qdrant.py --port 8000

    # Run the agent as usual:
    cargo run --bin gw-bench ... --search-url http://localhost:8000
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
from searchers.base import EncoderClient
from searchers.qdrant_searcher import QdrantSearcher


class QdrantSearchHandler(BaseHTTPRequestHandler):
    qdrant_searcher: QdrantSearcher = None
    corpus_texts: dict = None  # docid → full text for get_document

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/call/search":
            query = body.get("query", "")
            k = body.get("k", 5)
            results = self.qdrant_searcher.search(query, k=k)
            out = []
            for r in results:
                text = self.corpus_texts.get(r.docid, "")
                out.append({
                    "docid": r.docid,
                    "score": r.score,
                    "snippet": text[:3000],
                })
            self._json_response(200, out)

        elif self.path == "/call/get_document":
            docid = body.get("docid", "")
            text = self.corpus_texts.get(str(docid))
            if text:
                self._json_response(200, text)
            else:
                self._json_response(404, {"error": f"docid {docid} not found"})

        elif self.path == "/":
            self._json_response(200, {"status": "ok", "backend": "qdrant"})

        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        self._json_response(200, {"status": "ok", "backend": "qdrant"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def load_corpus(path: str) -> dict[str, str]:
    """Load docid → full text mapping from corpus_meta.jsonl."""
    print(f"Loading corpus text: {path}", flush=True)
    texts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts[str(obj["docid"])] = obj["text"]
    print(f"  loaded {len(texts)} docs", flush=True)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="colbert_mv")
    parser.add_argument("--encoder-url", default="http://127.0.0.1:8002")
    parser.add_argument("--corpus",
                        default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    args = parser.parse_args()

    # Load corpus for get_document
    QdrantSearchHandler.corpus_texts = load_corpus(args.corpus)

    # Init Qdrant searcher
    encoder = EncoderClient(args.encoder_url)
    QdrantSearchHandler.qdrant_searcher = QdrantSearcher(
        encoder, qdrant_url=args.qdrant_url, collection=args.collection,
    )

    server = HTTPServer((args.host, args.port), QdrantSearchHandler)
    print(f"Qdrant search server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
