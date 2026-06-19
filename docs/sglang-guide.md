# Running with SGLang

SGLang is an alternative LLM inference backend that offers significantly faster
multi-turn performance than Ollama through automatic prefix caching
(RadixAttention). In our BrowseComp agent loop, where each turn resends the
full conversation history, this avoids reprocessing shared prefixes and can
reduce per-query latency by 3-5x.

## Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 4090, 24GB VRAM)
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- HuggingFace model access (auto-downloaded on first run)

## Quick Start (Docker)

```bash
# Start SGLang server (first run downloads ~18GB model)
docker compose -f docker/docker-compose.yml up sglang

# Wait for health check to pass (1-2 minutes for model loading)
curl http://localhost:30000/health
```

The SGLang service is defined in `docker/docker-compose.yml` and runs
`Qwen/Qwen3.5-9B` on port 30000 by default.

## Running Benchmarks

### Python client (ollama_client.py)

```bash
python bench/browsecomp/ollama_client.py \
    --backend sglang \
    --model Qwen/Qwen3.5-9B \
    --searcher-type bm25s \
    --index-path data/bm25s-index \
    --max-turns 12
```

### Rust benchmark (gw-bench)

```bash
cargo run --bin gw-bench -- \
    --llm-backend sglang \
    --llm-url http://localhost:30000 \
    --model Qwen/Qwen3.5-9B \
    --search-backend native \
    --tantivy-index data/tantivy-corpus \
    --max-turns 12
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GW_LLM_BACKEND` | `ollama` | Backend: `ollama` or `sglang` |
| `GW_LLM_URL` | (per-backend) | LLM server URL |
| `SGLANG_URL` | `http://localhost:30000` | SGLang server URL (Python fallback) |
| `GW_MODEL` | `qwen2.5:7b` | Model name |

## Running without Docker

```bash
pip install "sglang[all]"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85
```

## GPU Memory Tuning

The `--mem-fraction-static` flag controls how much VRAM is reserved for the
KV cache. The default (0.85) works well on a 24GB RTX 4090 for the 9B model
at fp16.

| GPU VRAM | Model | Recommended `--mem-fraction-static` |
|----------|-------|-------------------------------------|
| 24GB (4090) | Qwen3.5-9B fp16 | 0.85 |
| 24GB (4090) | Qwen3.5-9B fp8 | 0.90 |
| 16GB (4080) | Qwen3.5-9B fp8 | 0.80 |

If you hit OOM, lower the fraction or add quantization:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --quantization fp8 \
    --mem-fraction-static 0.80 \
    ...
```

## Model Names

SGLang uses HuggingFace model paths, not Ollama tags:

| Ollama tag | HuggingFace path |
|------------|-----------------|
| `qwen3.5:9b` | `Qwen/Qwen3.5-9B` |
| `qwen2.5:7b` | `Qwen/Qwen2.5-7B-Instruct` |

The `--model` flag passed to the benchmark must match the model loaded
by the SGLang server.

## Embeddings

SGLang handles chat inference only. Embeddings (`nomic-embed-text` for LanceDB
vector search) still go through Ollama. When using `gw-bench` with
`--llm-backend sglang`, the Rust client routes:

- Chat requests → SGLang (`--llm-url`)
- Embedding requests → Ollama (`--ollama-url`)

Make sure Ollama is running if your benchmark uses vector search or LanceDB
index building.

## Verifying Prefix Caching

SGLang enables RadixAttention by default. To confirm it's working, check
the server logs for cache hit rates, or compare `prompt_tokens` across turns.
With prefix caching active, later turns should report significantly fewer
prompt tokens than the full conversation length.

## Troubleshooting

**Server won't start / OOM**: Lower `--mem-fraction-static` to 0.75 or add
`--quantization fp8`.

**Slow first request**: SGLang compiles CUDA kernels on first inference.
Subsequent requests are fast.

**Model not found**: Ensure the HuggingFace model path is correct. For gated
models, set `HF_TOKEN` in the Docker environment or shell.

**Health check failing**: Model loading can take 1-2 minutes. The Docker
healthcheck has a 2-minute start period. Check logs with
`docker compose logs sglang`.
