# Design: SGLang Inference Backend

**Status:** Proposal
**Date:** 2026-03-22

## Motivation

Our BrowseComp agent loop sends 12+ sequential LLM calls per question, each
resending the full conversation history. Ollama recomputes the entire prompt
from scratch on every call. With cumulative input tokens reaching 150K+ per
query, this is the dominant bottleneck.

SGLang's RadixAttention automatically caches KV state for shared prefixes
across turns — meaning turns 2-12 only process *new* tokens, not the full
history. Published benchmarks show 5-13x throughput improvement over Ollama
for multi-turn workloads. This directly addresses our biggest performance
constraint.

## Current Ollama Integration Surface

### Python (`bench/browsecomp/`)

| Component | Endpoint | Details |
|-----------|----------|---------|
| `ollama_client.py` — agent chat | `POST /api/chat` | `{"model", "messages", "stream": false}` → `message.content`, `prompt_eval_count`, `eval_count`. Timeout: 300s |
| `ollama_client.py` — `llm_query()` | `POST /api/chat` | Same endpoint, 60s timeout, input truncated to 8000 chars |
| `lancedb_searcher.py` — embeddings | `POST /api/embed` | `{"model", "input": [...]}` → `embeddings[]`. Batch size 32, timeout 120s |

### Rust (`crates/gw-llm/src/lib.rs`)

| Method | Endpoint | Details |
|--------|----------|---------|
| `chat()` | `POST /api/chat` | Non-streaming. Parses `message.content`, `prompt_eval_count`, `eval_count`. Optional `think` field. Timeout: 300s |
| `chat_stream()` | `POST /api/chat` | `stream: true`, NDJSON lines with `message.content` + `done` flag |
| `embed()` | `POST /api/embed` | Batch of 32, 8192 char max per text. Retry with 4096 on failure. Zero-vector fallback |

### Rust (`crates/gw-bench/src/main.rs`)

- Instantiates `OllamaClient` with `(proxy_url, direct_url, model, embedding_model)`
- Aggregates `input_tokens` + `output_tokens` into `UsageInfo` per query

### Configuration

- `config/greatwheel.toml`: `[llm]` section with `proxy_url`, `ollama_url`, `default_model`, `embedding_model`
- Docker Compose: `ollama/ollama` service on port 11434
- Env vars: `OLLAMA_URL`, `GW_MODEL`, `GW_EMBED_MODEL`

## SGLang Capabilities

### Why SGLang over vLLM

Both offer continuous batching, PagedAttention, and OpenAI-compatible APIs.
SGLang's advantage is **RadixAttention** — a radix-tree-based KV cache that
automatically detects and reuses shared prefixes across requests. This is
purpose-built for our multi-turn agent loop pattern. Published benchmarks
show SGLang 1.3-2x faster than vLLM for multi-turn workloads specifically.

### Key features

- **RadixAttention**: Automatic prefix caching via radix tree. No configuration needed — the runtime detects shared prefixes and reuses cached KV state. LRU eviction under memory pressure.
- **Continuous batching**: Dynamic batch sizing for GPU utilization.
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/embeddings` — standard request/response format.
- **Qwen 3.5 support**: Full support including reasoning (`--reasoning-parser qwen3`) and tool calling.
- **Quantization**: GPTQ, AWQ, FP8, INT8, GGUF, bitsandbytes. Offline quantization recommended.
- **Multi-GPU**: `--tp N` for tensor parallelism, `--dp N` for data parallelism.

### API mapping

**Chat:**
```
Ollama:  POST /api/chat     {"model", "messages", "stream"}
SGLang:  POST /v1/chat/completions  {"model", "messages", "stream"}
```

**Embeddings:**
```
Ollama:  POST /api/embed     {"model", "input": [...]}  → {"embeddings": [[...]]}
SGLang:  POST /v1/embeddings {"model", "input": [...]}  → {"data": [{"embedding": [...]}]}
```

**Token counting:**
```
Ollama:  prompt_eval_count, eval_count
SGLang:  usage.prompt_tokens, usage.completion_tokens
```

## Design

### Approach: Backend abstraction with provider trait

Rather than replacing Ollama, add SGLang as a second backend behind a common
interface. This lets us:
- A/B test performance between backends
- Keep Ollama for quick local dev (no HuggingFace download needed)
- Support future backends (vLLM, TGI) with minimal effort

### Rust changes (`gw-llm`)

Add an `LlmBackend` enum and adapt `OllamaClient` into a trait-based design:

```rust
pub enum LlmBackend {
    Ollama,
    Sglang,
}

pub struct LlmClient {
    backend: LlmBackend,
    base_url: String,
    model: String,
    embedding_model: String,
    http: reqwest::Client,
}
```

The client translates requests/responses based on `backend`:

| Operation | Ollama path | SGLang path |
|-----------|-------------|-------------|
| Chat | `/api/chat` | `/v1/chat/completions` |
| Embed | `/api/embed` | `/v1/embeddings` |
| Token count response field | `prompt_eval_count` / `eval_count` | `usage.prompt_tokens` / `usage.completion_tokens` |
| Stream format | NDJSON `{"message":{"content":"..."}, "done": bool}` | SSE `data: {"choices":[{"delta":{"content":"..."}}]}` |
| Think/reasoning | `"think": true` in body | `--reasoning-parser qwen3` server-side; `reasoning_content` in response |

### Python changes (`bench/browsecomp/`)

Add a `--backend` flag (`ollama` or `sglang`) to `ollama_client.py` and
`lancedb_searcher.py`. The flag controls:

1. **URL construction**: `/api/chat` vs `/v1/chat/completions`
2. **Payload format**: Ollama-native vs OpenAI-compatible
3. **Response parsing**: Different token count field names, embedding response shape
4. **Timeout defaults**: Can keep the same values initially

Minimal refactor — a thin adapter layer, not a rewrite:

```python
class LLMClient:
    def __init__(self, base_url, model, backend="ollama"):
        self.backend = backend
        # ...

    def _chat(self, messages):
        if self.backend == "sglang":
            return self._chat_openai(messages)
        return self._chat_ollama(messages)
```

### Configuration changes

**`config/greatwheel.toml`:**
```toml
[llm]
backend = "sglang"              # "ollama" | "sglang"
url = "http://localhost:30000"   # SGLang default port
default_model = "Qwen/Qwen3.5-9B"
embedding_model = "nomic-embed-text"
```

**CLI args** (gw-bench, Python scripts):
```
--backend sglang --llm-url http://localhost:30000
```

**Environment variables:**
```
GW_LLM_BACKEND=sglang
GW_LLM_URL=http://localhost:30000
```

### Docker Compose

Add an `sglang` service alongside (not replacing) Ollama:

```yaml
sglang:
  image: lmsysorg/sglang:latest
  command: >
    python3 -m sglang.launch_server
    --model-path Qwen/Qwen3.5-9B
    --host 0.0.0.0
    --port 30000
    --mem-fraction-static 0.85
    --watchdog-timeout 1200
  ports:
    - "30000:30000"
  volumes:
    - hf-cache:/root/.cache/huggingface
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
```

### Embedding model considerations

SGLang supports embedding via `/v1/embeddings` but requires compatible models
(e5-mistral, gte-Qwen2, etc). Our current `nomic-embed-text` model works with
Ollama but may not be directly loadable in SGLang.

Options:
1. **Keep Ollama for embeddings only** — SGLang for chat, Ollama for `/api/embed`. Simplest path.
2. **Switch embedding model** — use `Alibaba-NLP/gte-Qwen2-7B-instruct` or similar on SGLang.
3. **Separate embedding server** — dedicated SGLang instance for embeddings.

Recommendation: Option 1 for initial rollout. Embeddings are called during
index building (not the hot path), so Ollama's speed is fine there.

## Implementation Plan

### Phase 1: Python benchmark backend (fastest impact)

1. Add `--backend sglang` flag to `ollama_client.py`
2. Implement OpenAI-compatible chat adapter (URL, payload, response parsing)
3. Keep Ollama for embeddings
4. Run BrowseComp benchmark with SGLang to measure speedup
5. **Expected outcome**: 3-10x faster per-query latency from prefix caching

### Phase 2: Rust `gw-llm` backend

1. Add `LlmBackend` enum and config parsing
2. Implement OpenAI-compatible chat path in `LlmClient`
3. Implement OpenAI-compatible streaming path (SSE vs NDJSON)
4. Update `gw-bench` to pass backend config through
5. Update `config/greatwheel.toml` schema

### Phase 3: Docker and deployment

1. Add `sglang` service to `docker-compose.yml`
2. Document HuggingFace model download and GPU requirements
3. Add health check endpoint monitoring

### Phase 4: Optimization

1. Tune `--mem-fraction-static`, `--chunked-prefill-size` for our GPU
2. Experiment with quantization (GPTQ-INT4 for faster inference)
3. Benchmark prefix cache hit rates across BrowseComp runs
4. Consider `--kv-cache-dtype fp8_e5m2` for memory savings

## Risks

| Risk | Mitigation |
|------|------------|
| Qwen 3.5 9B not available on HuggingFace in same quantization as Ollama | Test with official Qwen HF checkpoints; fall back to GPTQ quants |
| SGLang startup time (~minutes for large models) | Use `keep_alive` equivalent; SGLang keeps models loaded by default |
| GPU memory: SGLang may reserve more VRAM than Ollama | Tune `--mem-fraction-static`; start conservative at 0.80 |
| `nomic-embed-text` not loadable in SGLang | Keep Ollama for embeddings (Phase 1 plan) |
| Response format drift across SGLang versions | Pin SGLang version in Docker image tag |

## Expected Impact

For a typical 12-turn BrowseComp query with ~150K cumulative input tokens:

| Metric | Ollama (current) | SGLang (projected) |
|--------|-------------------|---------------------|
| Prefix cache reuse | None | ~50-70% of prompt_eval skipped |
| Per-query wall time | ~60-90s | ~15-30s |
| Throughput (concurrent) | 1 req at a time | Continuous batching, N concurrent |
| Total benchmark (30 questions) | ~30-45 min | ~8-15 min |

The prefix caching benefit compounds: each additional turn reuses more cached
state, so the later (and longest) turns see the biggest speedup.
