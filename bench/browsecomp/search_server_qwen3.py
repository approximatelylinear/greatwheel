"""Search server wrapping the Qwen3-Embedding dense retriever for the agent loop.

Drop-in replacement for search_server.py — same HTTP API contract
(POST /call/search, POST /call/get_document) so the agent loop doesn't
need any changes.

Requires:
  1. The qwen3 LanceDB index built (build_qwen3_index.py)
  2. The Qwen3-Embedding encoder service running (qwen3_embed_server.py)
  3. The corpus jsonl for get_document text lookups

Usage:
    # Encoder service (long-lived):
    python bench/browsecomp/qwen3_embed_server.py --port 8003 &

    # This server:
    uv run --project bench/browsecomp --extra qwen3-embed \\
        python bench/browsecomp/search_server_qwen3.py --port 8000

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from searchers.qwen3_searcher import Qwen3Searcher


class Qwen3SearchHandler(BaseHTTPRequestHandler):
    searcher: Qwen3Searcher = None
    corpus_texts: dict = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/call/search":
            query = body.get("query", "")
            k = body.get("k", 5)
            results = self.searcher.search(query, k=k)
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
            self._json_response(200, {"status": "ok", "backend": "qwen3"})

        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        self._json_response(200, {"status": "ok", "backend": "qwen3"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def load_corpus(path: str) -> dict[str, str]:
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
    parser.add_argument("--encoder-url", default="http://127.0.0.1:8003")
    parser.add_argument("--index-dir", default="data/qwen3-embed")
    parser.add_argument("--corpus",
                        default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    args = parser.parse_args()

    Qwen3SearchHandler.corpus_texts = load_corpus(args.corpus)
    Qwen3SearchHandler.searcher = Qwen3Searcher(
        index_dir=args.index_dir,
        encoder_url=args.encoder_url,
    )

    server = HTTPServer((args.host, args.port), Qwen3SearchHandler)
    print(f"Qwen3 search server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
