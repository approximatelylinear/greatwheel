"""monoT5-3B cross-encoder rerank server.

Drop-in for rerank_server_blobs.py — same HTTP contract on POST /rerank.
Scores each (query, document) pair with castorini/monot5-3b-msmarco-10k
(or a configurable variant) and returns the top-k.

Meng et al. 2026 (arXiv:2602.21456) report this as the largest single
lever on BrowseComp-Plus: BM25 + monoT5-3B at depth 50 lifts gpt-oss-20b
accuracy from 0.572 to 0.689 (+20.5% relative).

Usage:
    uv run --project bench/browsecomp --extra monot5 \\
        python bench/browsecomp/rerank_server_monot5.py --port 8002

    target/release/gw-bench \\
        --search-backend native \\
        --tantivy-index data/tantivy-corpus/ \\
        --passage-index data/passages-4096/ \\
        --rerank-url http://localhost:8002 \\
        --model qwen3.5:9b \\
        --config bench/browsecomp/configs/baseline.toml \\
        --query bench/browsecomp/sample30.tsv \\
        --output-dir runs/m1-monot5 \\
        --k 10 --max-turns 12
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer


class MonoT5Reranker:
    """Cross-encoder reranker built on monoT5.

    Scores (query, doc) pairs by reading the decoder's first-step logits at
    the "▁true" / "▁false" tokens and taking softmax(true) as the score.
    """

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 16, max_length: int = 512):
        print(f"Loading monoT5 model: {model_name}", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Locate the token IDs for "true" and "false" — monoT5 was trained to
        # emit one of these as the decoder's first generated token.
        self.true_id = self.tokenizer.convert_tokens_to_ids("▁true")
        self.false_id = self.tokenizer.convert_tokens_to_ids("▁false")
        if self.true_id == self.tokenizer.unk_token_id or self.false_id == self.tokenizer.unk_token_id:
            # Some monoT5 variants use plain "true" / "false" without the SentencePiece prefix.
            self.true_id = self.tokenizer.convert_tokens_to_ids("true")
            self.false_id = self.tokenizer.convert_tokens_to_ids("false")
        print(f"  true_id={self.true_id} false_id={self.false_id}", flush=True)
        print(f"  device={device} dtype={dtype} batch_size={batch_size} max_length={max_length}", flush=True)

    @torch.inference_mode()
    def score_batch(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        inputs = [f"Query: {query} Document: {d} Relevant:" for d in docs]
        scores: list[float] = []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            # Decoder input is the BOS-equivalent for T5 (pad token).
            decoder_input_ids = torch.full(
                (enc["input_ids"].size(0), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
            # Logits at first decoder step: shape (batch, 1, vocab).
            logits = outputs.logits[:, 0, :]
            # Restrict softmax to {true, false}.
            stacked = torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1)
            probs = F.softmax(stacked, dim=1)[:, 1]  # P(true)
            scores.extend(probs.float().tolist())
        return scores

    def rerank(self, query: str, documents: list[dict], k: int) -> list[dict]:
        texts = [d.get("text", "") for d in documents]
        scores = self.score_batch(query, texts)
        ranked = sorted(zip(documents, scores), key=lambda p: p[1], reverse=True)
        out = []
        for d, s in ranked[:k]:
            out.append({
                "docid": d["docid"],
                "score": float(s),
                "text": d.get("text", ""),
            })
        return out


class MonoT5Handler(BaseHTTPRequestHandler):
    reranker: MonoT5Reranker = None

    def log_message(self, fmt, *args):
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

            t0 = time.monotonic()
            reranked = self.reranker.rerank(query, documents, k)
            dur_ms = int((time.monotonic() - t0) * 1000)
            print(f"[rerank] q={query[:40]!r} n={len(documents)} k={k} took={dur_ms}ms", flush=True)
            self._json_response(200, reranked)
        elif self.path == "/":
            self._json_response(200, {"status": "ok", "backend": "monot5"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        self._json_response(200, {"status": "ok", "backend": "monot5"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="castorini/monot5-3b-msmarco-10k")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    MonoT5Handler.reranker = MonoT5Reranker(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    server = HTTPServer(("127.0.0.1", args.port), MonoT5Handler)
    print(f"monoT5 rerank server listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
