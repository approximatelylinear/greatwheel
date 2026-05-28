"""Modal deployment of the Qwen3-Embedding server with an OpenAI-compatible
`/v1/embeddings` endpoint.

This reuses the validated Qwen3-Embedding recipe from
`bench/browsecomp/qwen3_embed_server.py` (left-padding + last-token/EOS
pooling + L2 normalization) but exposes it over the standard OpenAI
embeddings protocol so `gw-llm`'s `OllamaClient::embed` can call it
directly when `backend = "sglang"` and `direct_url` points here.

Cost model: scales to zero when idle (`scaledown_window`), so you only
pay for GPU seconds during actual requests. First request after idle
pays a cold start (model load to GPU) — accepted for the private demo.

Deploy:
    modal deploy deploy/modal/embed_server.py

The deployed URL (printed on deploy) is what you set as `direct_url`.
Auth: set a shared bearer token in a Modal secret named `gw-embed-auth`
with key `GW_EMBED_TOKEN`, and give the same token to `gw-server` as the
client API key:
    modal secret create gw-embed-auth GW_EMBED_TOKEN=$(openssl rand -hex 32)
"""

import modal

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)


def _download_model() -> None:
    """Bake the weights into the image so cold starts skip the network fetch."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "huggingface_hub",
        "fastapi[standard]",
    )
    .run_function(_download_model)
)

app = modal.App("gw-embed")


@app.cls(
    gpu="A10G",
    image=image,
    min_containers=0,       # scale to zero when idle
    scaledown_window=300,   # wait 5 min of inactivity before stopping
    secrets=[modal.Secret.from_name("gw-embed-auth")],
)
class Embedder:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModel, AutoTokenizer

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.token = os.environ["GW_EMBED_TOKEN"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
            .to(self.device)
            .eval()
        )

    def _encode(self, texts, max_length=4096, instruction="", sub_batch=8):
        import torch

        if instruction:
            texts = [instruction + t for t in texts]
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), sub_batch):
                batch = texts[start : start + sub_batch]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"]
                # last non-pad token per row (works with left padding)
                left_padded = mask[:, -1].sum() == mask.shape[0]
                if left_padded:
                    pooled = hidden[:, -1]
                else:
                    seq_lens = mask.sum(dim=1) - 1
                    idx = torch.arange(hidden.shape[0], device=hidden.device)
                    pooled = hidden[idx, seq_lens]
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.extend(normed.float().cpu().tolist())
        return out

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel

        api = FastAPI()

        class EmbeddingsRequest(BaseModel):
            model: str | None = None
            input: list[str] | str

        def _check_auth(authorization: str | None):
            expected = f"Bearer {self.token}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="invalid token")

        @api.get("/")
        def health():
            return {"status": "ok", "model": MODEL_NAME, "device": self.device}

        @api.post("/v1/embeddings")
        def embeddings(req: EmbeddingsRequest, authorization: str = Header(None)):
            _check_auth(authorization)
            texts = [req.input] if isinstance(req.input, str) else req.input
            # Doc-mode (no instruction prefix): matches the corpus side of the
            # recipe and is the right default for clustering/UMAP. Retrieval
            # queries that need the instruction prefix are a separate concern.
            vectors = self._encode(texts, instruction="")
            data = [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ]
            return {
                "object": "list",
                "data": data,
                "model": req.model or MODEL_NAME,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        return api
