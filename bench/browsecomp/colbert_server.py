"""Shared HTTP encoder service for the ColBERT benchmark backends.

Loads Reason-ModernColBERT once and serves token vectors over HTTP. Every
backend (Qdrant, LanceDB MV, Elasticsearch, brute-force, BM25→rerank, …)
hits this instead of loading its own encoder, so the model is on the GPU
once and shared across all benchmark + index-building workloads.

Endpoints
---------

POST /encode_query
    body:  {"text": "query string"}
    reply: {"tokens": [[f32; 128], ...], "n_tokens": int}

POST /encode_query_batch
    body:  {"texts": ["q1", "q2", ...]}
    reply: {"results": [{"tokens": [...], "n_tokens": int}, ...]}

POST /encode_doc_batch
    body:  {"texts": ["doc1", "doc2", ...], "max_length": 512}
    reply: {"results": [{"tokens": [...], "n_tokens": int}, ...]}

GET  /
    reply: {"status": "ok", "model": "...", "device": "cuda|cpu"}

Usage
-----

    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/colbert_server.py \\
        --model lightonai/Reason-ModernColBERT --port 8002

Notes
-----

* Single-process, single-thread (HTTP handlers serialize on the encoder).
  That's fine for benchmarks — model loads, queries are batched per call,
  and we only have a few concurrent backends.
* Returns plain Python floats so any HTTP client (Rust, JS, Python) can
  decode without numpy.
* Backwards compat: the legacy `/encode` endpoint is still served as an
  alias for `/encode_query`.
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
from colbert_encode import ColBERTEncoder


class EncodeHandler(BaseHTTPRequestHandler):
    encoder: ColBERTEncoder = None
    model_name: str = ""

    def log_message(self, format, *args):  # silence access log
        pass

    # ---- entry points ----------------------------------------------------

    def do_GET(self):
        if self.path == "/":
            self._json_response(200, {
                "status": "ok",
                "model": self.model_name,
                "device": str(self.encoder.device),
            })
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"bad json: {e}"})
            return

        if self.path in ("/encode_query", "/encode"):
            self._handle_encode_query(body)
        elif self.path == "/encode_query_batch":
            self._handle_encode_query_batch(body)
        elif self.path == "/encode_doc_batch":
            self._handle_encode_doc_batch(body)
        else:
            self._json_response(404, {"error": "not found"})

    # ---- handlers --------------------------------------------------------

    def _handle_encode_query(self, body):
        text = body.get("text", "")
        if not text:
            self._json_response(400, {"error": "missing 'text' field"})
            return
        tokens = self.encoder.encode_query(text)
        self._json_response(200, {"tokens": tokens, "n_tokens": len(tokens)})

    def _handle_encode_query_batch(self, body):
        texts = body.get("texts")
        if not isinstance(texts, list) or not texts:
            self._json_response(400, {"error": "missing or empty 'texts' list"})
            return
        # ColBERTEncoder.encode_query is single-text, but _encode handles batches.
        tensors = self.encoder._encode(texts, max_length=128, is_query=True)
        results = []
        for t in tensors:
            tokens = t.tolist()
            results.append({"tokens": tokens, "n_tokens": len(tokens)})
        self._json_response(200, {"results": results})

    def _handle_encode_doc_batch(self, body):
        texts = body.get("texts")
        if not isinstance(texts, list) or not texts:
            self._json_response(400, {"error": "missing or empty 'texts' list"})
            return
        max_length = int(body.get("max_length", 512))
        tensors = self.encoder._encode(texts, max_length=max_length, is_query=False)
        results = []
        for t in tensors:
            tokens = t.tolist()
            results.append({"tokens": tokens, "n_tokens": len(tokens)})
        self._json_response(200, {"results": results})

    # ---- util ------------------------------------------------------------

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
    EncodeHandler.model_name = args.model
    print(f"  device: {EncodeHandler.encoder.device}", flush=True)

    server = HTTPServer((args.host, args.port), EncodeHandler)
    print(f"ColBERT encode server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
