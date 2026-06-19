# Qwen3-Embedding Retrieval Experiment

## Hypothesis

Our prior dense-retrieval result ("vector search doesn't help on BrowseComp,
regardless of embedding model" — EXPERIMENTS.md, R1) was tested only with
nomic-embed-text-v1.5 (~137M encoder). The BrowseComp-Plus paper shows
Qwen3-Embedding-8B clearly outperforming BM25 across agent classes
(SearchR1-32B, Gemini 2.5 Pro, Opus 4, gpt-oss-120B, o3, GPT-5).

The conjecture: the gap is the **embedder class**, not the retrieval
paradigm. Qwen3-Embedding is a decoder-LLM-based embedder finetuned for
retrieval — large enough to read and reason over a full document into one
vector. nomic is too small to do that on entity-heavy 100K-doc corpora.

This experiment isolates that hypothesis by mirroring the BrowseComp-Plus
indexing recipe exactly, starting with the smallest Qwen3-Embedding (0.6B)
to keep build/iteration cost low.

## Configuration we mirror from BrowseComp-Plus

Source: `vendor/BrowseComp-Plus/scripts_build_index/qwen3-embed.md` and the
matching Tevatron example.

| Setting              | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Model                | `Qwen/Qwen3-Embedding-0.6B` (we'll escalate to 4B/8B if 0.6B helps)    |
| Dimensionality       | 1024 (0.6B), 2560 (4B), 4096 (8B) — read from model config             |
| Pooling              | EOS / last-non-pad-token                                               |
| Normalization        | L2                                                                     |
| Precision            | fp16                                                                   |
| **Doc** max length   | 4096 tokens                                                            |
| **Doc** prefix       | none                                                                   |
| **Doc** granularity  | one vector per full document (NOT chunked into passages)               |
| **Query** max length | 512 tokens                                                             |
| **Query** prefix     | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` |
| Retrieval            | single-vector cosine (dot product over normalized vectors), top-k      |

The full-doc + EOS pooling choice is important: chunking into passages
defeats the LLM-embedder's ability to summarize a long doc into one vector.
That's the regime BC+ benchmarks, and the regime nomic was *not* tested
under here.

## Components

| File                                          | Role                                                                |
|-----------------------------------------------|---------------------------------------------------------------------|
| `qwen3_embed_server.py`                       | HTTP encode-only service, port 8003. Loads model once.              |
| `build_qwen3_index.py`                        | Streams corpus, encodes via service, writes LanceDB single-vector.  |
| `searchers/qwen3_searcher.py`                 | `Searcher` impl: encode query → LanceDB cosine top-k.               |
| `retrieval_benchmark_v2.py` (`qwen3` backend) | Wires it into the multi-backend R@k harness.                        |

The service runs separately from the existing ColBERT encoder service
(port 8002 vs 8003) — different model, different output shape. They can
coexist on the same GPU if VRAM permits, or be swapped between runs.

## Run plan

### Phase 1 — Retrieval R@k

```bash
# (one-time) install deps
uv sync --project bench/browsecomp --extra qwen3-embed

# 1. Encoder service (long-lived; tail logs)
uv run --project bench/browsecomp --extra qwen3-embed \
    python bench/browsecomp/qwen3_embed_server.py --port 8003

# 2. Smoke test on 1000 docs
uv run --project bench/browsecomp --extra qwen3-embed \
    python -u bench/browsecomp/build_qwen3_index.py --max-docs 1000

# 3. Full corpus build (~100K docs)
uv run --project bench/browsecomp --extra qwen3-embed \
    python -u bench/browsecomp/build_qwen3_index.py

# 4. Compare against existing backends
uv run --project bench/browsecomp --extra qwen3-embed \
    python bench/browsecomp/retrieval_benchmark_v2.py \
    --searchers qwen3 tantivy qdrant lancedb_mv \
    --json-out runs/qwen3_vs_baselines.json
```

**Decision gate:** if Qwen3-0.6B R@200 on sample30 reaches ~20/30+
(BM25 is 12/30, ColBERT is 25/30), the embedder-class hypothesis is
confirmed and we proceed to Phase 2. If it stays near BM25, we have to
decide whether to escalate to 4B/8B (more compute but proves the chart's
exact regime) or shelve dense retrieval for this benchmark.

### Phase 2 — End-to-end agent eval

Wrap the searcher behind a `search_server_qwen3.py` (mirror of
`search_server_qdrant.py`) so the existing `ollama_client.py` agent can
call it as a drop-in replacement for the BM25 backend. Run sample30 with
qwen3.5:9b under the same baseline prompt used for prior end-to-end
comparisons. Compare against:

- BM25 baseline (9/30 historical, 5/30 recent reproduction)
- BM25 + blob rerank (7/30)
- Qdrant ColBERT (7/30)

If Qwen3-0.6B retrieval lifts R@200 substantially but agent accuracy
stalls or regresses, that's the same agent-co-adaptation bottleneck we
saw with ColBERT — escalate via R6 (GEPA on the new backend).
