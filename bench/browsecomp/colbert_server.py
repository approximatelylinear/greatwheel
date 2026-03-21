"""Lightweight HTTP server for ColBERT query encoding.

Returns 128-dim token embeddings for a query string. No search logic —
the Rust side handles LanceDB multi-vector search and BM25+ColBERT fusion.

Usage:
    uv run --project bench/browsecomp --extra colbert python bench/browsecomp/colbert_server.py \
        --model lightonai/Reason-ModernColBERT --port 8002

Endpoint:
    POST /encode  {"text": "query string"}
                  -> {"tokens": [[f32, ...], ...], "n_tokens": 7}
    GET  /        -> {"status": "ok"}
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
from colbert_encode import ColBERTEncoder


class EncodeHandler(BaseHTTPRequestHandler):
    encoder: ColBERTEncoder = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/encode":
            text = body.get("text", "")
            if not text:
                self._json_response(400, {"error": "missing 'text' field"})
                return

            tokens = self.encoder.encode_query(text)
            self._json_response(200, {
                "tokens": tokens,
                "n_tokens": len(tokens),
            })

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
    parser = argparse.ArgumentParser(description="ColBERT encode-only server")
    parser.add_argument("--model", default="lightonai/Reason-ModernColBERT",
                        help="ColBERT model name")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Loading ColBERT model: {args.model} ...", flush=True)
    EncodeHandler.encoder = ColBERTEncoder(args.model)

    server = HTTPServer((args.host, args.port), EncodeHandler)
    print(f"ColBERT encode server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
