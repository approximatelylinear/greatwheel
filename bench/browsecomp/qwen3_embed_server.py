"""Shared HTTP encoder service for Qwen3-Embedding.

Mirrors `colbert_server.py` but for single-vector dense embeddings. Loads
Qwen3-Embedding-{0.6B,4B,8B} once and serves L2-normalized pooled
embeddings over HTTP. The query endpoint applies the BrowseComp-Plus
instruction prefix; the doc endpoint applies no prefix. EOS pooling
(last non-pad token) matches the official Qwen3-Embedding recipe.

Endpoints
---------

POST /encode_query
    body:  {"text": "query string"}
    reply: {"vector": [f32; dim], "dim": int}

POST /encode_query_batch
    body:  {"texts": ["q1", "q2", ...]}
    reply: {"vectors": [[f32; dim], ...], "dim": int}

POST /encode_doc_batch
    body:  {"texts": ["doc1", ...], "max_length": 4096}
    reply: {"vectors": [[f32; dim], ...], "dim": int}

GET  /
    reply: {"status": "ok", "model": "...", "device": "...", "dim": int}

Usage
-----

    uv run --project bench/browsecomp --extra qwen3-embed \\
        python bench/browsecomp/qwen3_embed_server.py \\
        --model Qwen/Qwen3-Embedding-0.6B --port 8003
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoModel, AutoTokenizer


QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the hidden state at the last non-pad position for each row.

    Works with either padding side, per the Qwen3-Embedding reference impl.
    """
    left_padded = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_idx, seq_lens]


class Qwen3Encoder:
    """Load Qwen3-Embedding and produce L2-normalized embeddings."""

    def __init__(self, model_name: str, device: str | None = None, dtype: str = "fp16"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left-padding so the EOS / last meaningful token is always at index -1.
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int,
        instruction: str = "",
        sub_batch: int = 8,
    ) -> list[list[float]]:
        if instruction:
            texts = [instruction + t for t in texts]
        out: list[list[float]] = []
        for start in range(0, len(texts), sub_batch):
            batch = texts[start:start + sub_batch]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            pooled = _last_token_pool(hidden, enc["attention_mask"])
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(normed.float().cpu().tolist())
        return out


class EncodeHandler(BaseHTTPRequestHandler):
    encoder: Qwen3Encoder = None
    query_max_length: int = 512
    doc_max_length: int = 4096
    doc_sub_batch: int = 8
    query_sub_batch: int = 32

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._json_response(200, {
                "status": "ok",
                "model": self.encoder.model_name,
                "device": str(self.encoder.device),
                "dim": self.encoder.dim,
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

        if self.path == "/encode_query":
            self._handle_encode_query(body)
        elif self.path == "/encode_query_batch":
            self._handle_encode_query_batch(body)
        elif self.path == "/encode_doc_batch":
            self._handle_encode_doc_batch(body)
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_encode_query(self, body):
        text = body.get("text", "")
        if not text:
            self._json_response(400, {"error": "missing 'text' field"})
            return
        vec = self.encoder.encode(
            [text],
            max_length=self.query_max_length,
            instruction=QUERY_INSTRUCTION,
            sub_batch=self.query_sub_batch,
        )[0]
        self._json_response(200, {"vector": vec, "dim": len(vec)})

    def _handle_encode_query_batch(self, body):
        texts = body.get("texts")
        if not isinstance(texts, list) or not texts:
            self._json_response(400, {"error": "missing or empty 'texts' list"})
            return
        vecs = self.encoder.encode(
            texts,
            max_length=self.query_max_length,
            instruction=QUERY_INSTRUCTION,
            sub_batch=self.query_sub_batch,
        )
        self._json_response(200, {"vectors": vecs, "dim": len(vecs[0]) if vecs else 0})

    def _handle_encode_doc_batch(self, body):
        texts = body.get("texts")
        if not isinstance(texts, list) or not texts:
            self._json_response(400, {"error": "missing or empty 'texts' list"})
            return
        max_length = int(body.get("max_length", self.doc_max_length))
        vecs = self.encoder.encode(
            texts,
            max_length=max_length,
            instruction="",
            sub_batch=self.doc_sub_batch,
        )
        self._json_response(200, {"vectors": vecs, "dim": len(vecs[0]) if vecs else 0})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Embedding encode-only server")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--doc-max-length", type=int, default=4096)
    parser.add_argument("--doc-sub-batch", type=int, default=8,
                        help="Doc encoding sub-batch (lower for 4B/8B models or longer max_length)")
    parser.add_argument("--query-sub-batch", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading {args.model} ({args.dtype}) ...", flush=True)
    EncodeHandler.encoder = Qwen3Encoder(args.model, dtype=args.dtype)
    EncodeHandler.query_max_length = args.query_max_length
    EncodeHandler.doc_max_length = args.doc_max_length
    EncodeHandler.doc_sub_batch = args.doc_sub_batch
    EncodeHandler.query_sub_batch = args.query_sub_batch
    print(f"  device: {EncodeHandler.encoder.device}", flush=True)
    print(f"  dim: {EncodeHandler.encoder.dim}", flush=True)
    print(f"  query_max_length: {args.query_max_length}", flush=True)
    print(f"  doc_max_length: {args.doc_max_length}", flush=True)

    server = HTTPServer((args.host, args.port), EncodeHandler)
    print(f"Qwen3 embed server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
