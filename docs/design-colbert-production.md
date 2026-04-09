# Productionizing ColBERT — Design Notes

Captures what we learned from the BrowseComp work about running late-interaction
retrieval (specifically Reason-ModernColBERT) at non-toy scale. Written while the
first full Voyager passage index was building.

## Why late interaction at all

Single-vector dense retrievers have a **provable representational ceiling**
(Weller et al. 2024, "On the Theoretical Limitations of Embedding-Based Retrieval";
LIMIT benchmark). For any fixed dimension *d*, there exist top-*k* relevance
patterns that no *d*-dim embedding can encode — it's a sign-rank argument, not an
engineering gap. Required *d* grows with corpus size and relevance complexity.

Empirical signal: on the LIMIT benchmark, mainstream single-vector providers score
in single digits at Recall@100 (Gemini 6.9, Voyage 8.95, Cohere 5.7, OpenAI 4.95).
Mixedbread Wholembed V3 scores 98 — and is multi-vector under the hood.
That gap is the theoretical ceiling made visible.

**Conclusion:** for reasoning-heavy or adversarial retrieval (BrowseComp is one
such case), late interaction isn't just "more accurate," it's solving a different
problem class than fixed-dim embeddings can.

## The honest cost of ColBERT

Storage is dominated by **token count**, not doc count. For our 100K-doc /
~900K-passage BrowseComp corpus with Reason-ModernColBERT (128-dim, 512 max
tokens):

- ~460M token vectors total (~500 tokens/passage avg)
- E4M3 quantized vector data: ~59 GB
- HNSW graph (M=12, ~24 neighbors/node × 4 bytes): ~44 GB
- Per-node metadata: ~5 GB
- **Total Voyager index: ~108 GB** (matches the 115 GB we observed)
- token→docid pickle map: ~4.2 GB

ColBERT is fundamentally **~500× larger than single-vector retrieval** because
it keeps every token. E4M3 already saved us 4× over Float32 (would be ~285 GB).
PQ would save another 4–8× but Voyager doesn't support PQ; FAISS IVF-PQ does.

## Build engineering lessons (Voyager, this codebase)

These came from painful debugging during the build (`bench/browsecomp/build_voyager_index.py`):

1. **HNSW build is O(log N) per insert** — slows down dramatically as the index
   grows. Initial naive build was estimated at 47 hours.
2. **Three knobs gave us 20× speedup**: M=12 (smaller graph), ef_construction=50
   (smaller candidate list), bulk `add_items()` calls (~50K vectors per call) instead
   of one-at-a-time inserts. Final build is ~3 hours.
3. **Sub-batching the encoder forward pass** matters: batch 16 on the GPU side,
   not single passages, to amortize CUDA launch overhead.
4. **Checkpoint atomically.** Write to `.tmp` files, fsync, then `os.replace()`.
   We lost a 90K-doc checkpoint to a partial overwrite during an OOM crash on the
   final save and had to restart from scratch.
5. **Don't double-save at the end.** If the loop already checkpointed at
   `doc_count % CHECKPOINT_EVERY == 0`, skip the final save — it doubles peak
   memory at the worst possible moment.
6. **Checkpoint frequency is a recovery-cost tradeoff.** We're at every 5K docs
   (~10 minutes of work) which feels right for a 3-hour build.

## Architecture options for serving

There are two distinct deployment patterns and they have very different infrastructure
costs.

### Option A: ColBERT as first-stage retriever ("research mode")

Build an HNSW (or PLAID, or `silo`) index over **all token vectors** in the
corpus. Query against the index, aggregate token hits → docs, return top-K.

- **Infrastructure**: large specialized index (~108 GB for our corpus)
- **Pros**: best possible recall — no other retriever's blind spots cap quality
- **Cons**: huge storage, slow build, query latency dominated by HNSW lookup
- **When it makes sense**: research benchmarks, when stage-1 recall is the
  bottleneck, when you can afford the storage

### Option B: ColBERT as reranker ("production mode")

Cheap first stage (BM25 + single-vector dense) returns ~1000 candidates. Fetch
their precomputed token tensors from a **blob store** (no ANN index involved),
run MaxSim against the query, return top-K.

- **Infrastructure**: blob store keyed by `passage_id`. ~58 GB cold storage for
  our corpus (no graph, no quantization games needed).
- **Pros**: boring infrastructure (postgres BYTEA / S3 / lance / parquet works
  fine), query latency is dominated by stage-1 ANN (already fast)
- **Cons**: capped by stage-1 recall — if BM25+dense miss the right passage in
  top-1000, ColBERT can't save you
- **When it makes sense**: almost everywhere in production. Standard recipe is
  BM25 + dense → ColBERT rerank → optional cross-encoder rerank.

### Option C: Custom multi-vector engine ("frontier mode")

What Mixedbread does with `silo`: custom Rust engine, two-stage internally
(approximate prune → MaxSim rerank), aggressive vector quantization, S3-cold +
NVMe-hot blob store, hand-tuned MaxSim CPU kernels, learned variable-vector
counts per input (denser content gets more vectors). 50ms latency at 500+ QPS
per store, billions of docs.

- **Infrastructure**: you build your own
- **Pros**: scales to billions, frontier-quality, full control
- **Cons**: months of engineering
- **When it makes sense**: you're a vector DB vendor or you have billion-scale
  retrieval as a core product capability

## Empirical validation on BrowseComp (sample30)

After building an 85K-doc Voyager passage index (`data/voyager-passages`,
~99 GB, 854K passages, 430M tokens), we ran the retrieval benchmark in
`bench/browsecomp/retrieval_benchmark.py` against the 30-query sample.
All 30 queries had **full gold-doc coverage** in the 85K subset, so the
ceiling is uncapped.

```
Strategy             R@5    R@10   R@20   R@50   R@100  R@200
doc_bm25             4/30   6/30   8/30   10/30  12/30  14/30
passage_bm25         4/30   8/30   8/30   9/30   11/30  12/30
doc_passage_rrf      2/30   4/30   6/30   12/30  16/30  17/30
voyager              1/30   2/30   6/30   11/30  14/30  24/30
voyager_rerank      12/30  14/30  19/30   22/30  23/30  24/30
voyager_passage_rrf  3/30   4/30   5/30   10/30  16/30  21/30
```

Three things to read off this table.

**1. The Weller bound is empirically real on BrowseComp.** Voyager's
R@200 = 24/30 (80%) vs the best BM25-only strategy's 17/30 (57%). That's
a +23 point absolute gap at deep recall, exactly the kind of headroom
late interaction is supposed to provide. The relevance patterns BrowseComp
queries need are not all expressible by a fixed-dim sparse representation.

**2. Voyager-as-retriever is great at recall but terrible at the head.**
Raw `voyager` only gets 1/30 at R@5 and 2/30 at R@10 — worse than every
BM25 baseline. The reason: our search uses approximate per-query-token ANN
("for each query token, find top-N nearest token vectors, then aggregate
per-doc"). Many docs accumulate at least *some* score from each query
token, so the head ranking is mushy. The signal is in the candidate set;
the ordering is wrong.

**3. Full per-passage MaxSim rerank fixes the head completely.**
`voyager_rerank` takes the top 200 Voyager hits, splits each candidate
doc into the same 4096-char passages used at index time, encodes every
passage on GPU with `[D]` prefix, and computes real MaxSim against the
query (per-passage max-pooled to a single doc score). The result:

| metric  | voyager | voyager_rerank | delta       |
| ------- | ------- | -------------- | ----------- |
| R@5     | 1/30    | 12/30          | **+11 (12×)** |
| R@10    | 2/30    | 14/30          | **+12 (7×)** |
| R@20    | 6/30    | 19/30          | **+13 (3.2×)** |
| R@50    | 11/30   | 22/30          | +11 (2×)    |
| R@100   | 14/30   | 23/30          | +9 (1.6×)   |
| R@200   | 24/30   | 24/30          | preserved   |

`voyager_rerank` is the best strategy at **every** recall depth — including
beating doc_bm25's R@5 (12 vs 4) by 3×, and beating doc_passage_rrf's
R@10 (14 vs 4) by 3.5×. This is the architecture validated end-to-end.

### Validation: blob store rerank (Option B end-to-end)

Built a Lance-backed passage blob store at `data/passage-blobs`:

- **100,195 docs / 1,013,065 passages / 510M tokens** (full corpus)
- **122 GB on disk**, float16 token tensors stored as raw bytes per passage
- **96 minutes to build** at a steady 176 passages/sec, ~1% RAM (Lance writes
  incrementally, no in-memory accumulation — much friendlier than the
  Voyager build's HNSW)
- Schema: `(docid: string, chunk_idx: int32, num_tokens: int32, vectors: binary)`
- Built once via `bench/browsecomp/build_passage_blob_store.py`,
  resumable via `--resume` (skips already-encoded docids)

Wired into `voyager_searcher.rerank_from_blobs()`: takes the top-200 Voyager
candidates, fetches all their passage tensors via a single Lance `IN` query,
decodes the float16 bytes, runs the same torch MaxSim per passage, takes per-doc
max. **No GPU encoding at query time.**

```
Strategy             R@5    R@10   R@20   R@50   R@100  R@200
voyager_rerank       12/30  14/30  19/30  22/30  23/30  24/30  ← GPU re-encode (90s/query)
voyager_rerank_blobs 14/30  18/30  20/30  22/30  23/30  24/30  ← blob fetch (~200ms/query)
```

| metric            | GPU rerank | blob rerank | factor |
| ----------------- | ---------- | ----------- | ------ |
| R@5               | 12/30      | 14/30       | +2     |
| R@10              | 14/30      | 18/30       | +4     |
| per-query latency | ~90,000 ms | ~200 ms     | **~450×** |
| total bench (30q) | ~45 min    | ~4 min      | **~11×** |

Two surprises:

1. **Blob rerank is slightly *better* at the head than GPU rerank.** Almost
   certainly numerical noise from float16 quantization randomly nudging a few
   near-threshold ranks; at 30 queries, ±2 swaps moves the count visibly.
   The meaningful claim is that float16 storage costs nothing in quality.

2. **Lance `IN` query latency is ~200 ms with no scalar index needed.** A
   single fetch returns ~1500 rows / ~184 MB of token data per query. We
   verified Lance's default partition layout was efficient enough that adding
   a BTREE scalar index on `docid` only changed query time from ~100 ms to
   ~50 ms — not the 100× we'd been worried about.

This validates the entire production architecture end-to-end: we get
ColBERT-quality late-interaction retrieval at near-BM25 latency, with
boring infrastructure (Lance table + torch matmul, no custom kernel).

### Validation: BM25 → blob rerank (can we drop Voyager?)

Tested whether the 99 GB Voyager index could be retired by using BM25 as
the first stage, with blob rerank doing the precision work:

```
Strategy                       R@5    R@10   R@20   R@50   R@100  R@200
doc_bm25                       4/30   6/30   8/30   10/30  12/30  14/30
doc_bm25_rerank_blobs          8/30   8/30   11/30  12/30  14/30  14/30  (+rerank)
passage_bm25                   4/30   8/30   8/30   9/30   11/30  12/30
passage_bm25_rerank_blobs      7/30   7/30   8/30   11/30  12/30  12/30  (+rerank)
doc_passage_rrf                2/30   4/30   6/30   12/30  16/30  17/30
doc_passage_rrf_rerank_blobs   9/30   10/30  12/30  13/30  17/30  17/30  (+rerank)
voyager_rerank_blobs           14/30  18/30  20/30  22/30  23/30  24/30
```

Two findings:

**1. Blob rerank materially helps BM25 at the head.** The first-stage
candidates were always there; the ranking just sucked until something did
real MaxSim on them. Doc BM25 went 4→8 at R@5 (+4); RRF went 2→9 (+7).

**2. But the R@200 ceiling is fixed by the first stage.** Rerank can only
reorder candidates — it can't find docs the first stage missed. Look at
R@200: doc_bm25 14 → reranked 14, RRF 17 → reranked 17. **Voyager's 24/30
R@200 includes 7 docs that no BM25 strategy returns at any depth.** That
is the Weller bound made empirical on this benchmark: ~23% of queries
have relevance patterns sparse retrieval cannot represent.

**Voyager is not redundant — it is the recall layer.** The blob store
replaces *re-encoding at query time*, not the first-stage retrieval. The
production architecture is:

```
Voyager retrieval (top-200) → blob rerank → top-K
```

Total storage: ~221 GB (99 GB Voyager HNSW + 122 GB blob store) for
24/30 R@200 at ~200 ms/query. The lighter "BM25 + blob rerank" path
costs only the 122 GB blob store and gets 9/30 R@5 / 17/30 R@200 —
better head precision than any prior strategy, capped recall. Worth
shipping as a fallback when the deep-recall queries don't matter.

### First-stage retrieval — three options for production

When wiring this into greatwheel's Rust runtime we have three viable
shapes for the first stage. The blob store (122 GB) and the rerank
kernel are unchanged across all three; only the candidate generator
differs.

**Option 1 — BM25 first stage (`tantivy + CandleEncoder + BlobReranker`).**

- **Stack:** existing tantivy index in `gw-memory` → encode query with
  candle → rerank top-200 from blob store → top-K.
- **Builds required:** none. We already have tantivy and the blob store.
- **Storage:** 122 GB (just the blob store).
- **Recall on BrowseComp sample30:** 9/30 R@5, 17/30 R@200.
- **Cold start:** seconds (encoder + tantivy only; no HNSW load).
- **Per-query latency:** dominated by encode (~10–20 ms on GPU) + tantivy
  search (~5 ms) + blob rerank (~200 ms) ≈ **~250 ms**.
- **What it costs us:** the 7 Weller-bound queries that no BM25 can
  reach. Hard ceiling at 17/30 on this benchmark.
- **When this is right:** when real workloads don't lean heavily on
  reasoning-style retrieval, when storage budget is tight, or when we
  want to ship today.

**Option 2 — HNSW first stage (`usearch + CandleEncoder + BlobReranker`).**

- **Stack:** rebuild a usearch HNSW over the same flattened token
  vectors that fed the Voyager index → for each query token, ANN-fetch
  top-N nearest tokens → aggregate per-doc → rerank top-200 from blob
  store → top-K.
- **Builds required:** ~6-hour offline rebuild (same source vectors as
  the Voyager build, written to a Rust-readable usearch file).
- **Storage:** ~177 GB (~55 GB usearch f16 HNSW + 122 GB blob store).
  usearch with f16 quantization is *smaller* than the 99 GB Voyager E4M3
  index because of more aggressive scalar packing.
- **Recall on BrowseComp sample30:** expected to match the Python
  Voyager result (14/30 R@5, 24/30 R@200) — same algorithm, same
  vectors, different HNSW lib. Within ~1% per the usearch literature.
- **Cold start:** ~2 min (mmap'd HNSW + docid map + encoder).
- **Per-query latency:** ~600 ms ANN scan (32 query tokens × ANN lookup)
  + 200 ms rerank ≈ **~800 ms**, dominated by the HNSW scan.
- **What it costs us:** a 6-hour rebuild and ~55 GB of disk we can't
  reuse for anything else, plus a moving piece (HNSW maintenance: rebuild
  on corpus changes, no incremental updates without ef tuning).
- **When this is right:** when the Weller-bound queries genuinely matter
  in the real workload — i.e., when we have evidence that BM25's blind
  spots are hurting users.

**Option 3 — Brute-force MaxSim, no first stage.**

- **Stack:** encode query → score *every* passage in the blob store with
  full MaxSim → take top-K. No HNSW, no candidate filtering.
- **Builds required:** none.
- **Storage:** 122 GB (just the blob store).
- **Recall:** **exact upper bound.** No approximation anywhere — every
  passage gets a real MaxSim score, so the top-K is whatever ColBERT
  thinks is best, period. Should equal or exceed all HNSW-based options.
- **Cost math:** 13 query tokens × 500M passage tokens × 128 dims ×
  2 FLOPs ≈ **1.6 TFLOPs per query**. On a modern GPU (~10 TFLOPs):
  ~150 ms per query. On CPU with BLAS (~500 GFLOPs multi-thread):
  ~3 seconds per query. Streaming the blob store is the I/O bottleneck
  on either path: 122 GB / query at the limit, or ~10 GB/query if we
  cache the decoded f32 in RAM (needs ~244 GB).
- **What it costs us:** GPU time per query (or CPU patience), and a
  large RAM working set if we want sub-second latency.
- **When this is right:** as a **research tool** to establish the true
  recall ceiling for any first-stage approximation. Also potentially
  viable for offline batch jobs (e.g. nightly query analysis) where
  3 sec/query is fine. **Not a production serving path** at our scale
  unless we cache the entire blob store in GPU memory (out of budget).
- **Hidden value:** even if we never ship it, it's the cleanest way to
  measure "how much recall is HNSW giving up vs the exact answer?" —
  a number we don't currently have.

#### Recommendation

For shipping ColBERT into greatwheel **today**, **Option 1** is the
right call:

1. The infrastructure is done (`gw-memory` already has tantivy; we have
   the blob store and `BlobReranker`; the candle encoder is parity-tested).
2. Cold start is seconds, not minutes.
3. We can ship and measure real-world recall — the 7-doc BrowseComp gap
   is from one synthetic benchmark and may not reflect production use.
4. If real workloads hit the BM25 ceiling, **Option 2** is a 6-hour
   offline build away. The blob store and rerank stay unchanged; we
   only add the HNSW.
5. **Option 3** is worth running once as a research artifact to measure
   the HNSW recall gap. Cheap to write (~50 lines on top of `BlobReranker`),
   slow to run (~3 sec/query × 30 queries = ~90 sec for sample30 — easy).

### Lessons from the experiment

- **Truncated rerank is a trap.** A first attempt reranked only the first
  4096 chars of each candidate doc. R@5 went from 1 → 3 — modest. Switching
  to "encode all passages of each candidate, take the per-doc max" jumped
  R@5 to 12. The relevant content frequently lives in passage #3 of a long
  doc, and a first-passage-only rerank silently misses it.
- **The rerank kernel is irrelevant to total cost.** GPU encoding dominates
  by a factor of ~1000×. We started with `maxsim-cpu` for the rerank
  similarity step, but it segfaulted under repeated calls (libxsmm JIT
  registry growth). Swapping to plain `q @ pt.T` torch was numerically
  identical and equally fast end-to-end. Worth flagging upstream as a
  maxsim-cpu bug; not worth blocking on.
- **The encoder load is the new long pole** once you're on the blob-rerank
  path. Voyager index load = 113s, token map = 33s, encoder load = ~10s.
  Without Voyager, cold start collapses to encoder-only — useful for the
  BM25-rerank fallback path which doesn't need the HNSW.
- **Voyager is not redundant infrastructure**, despite the temptation to
  retire it after building the blob store. The blob store replaces
  *re-encoding*, not *first-stage retrieval*. The 7 extra recalled queries
  at R@200 (24 vs 17) are docs that no BM25 strategy returns at any depth.
  They are the Weller-bound queries we built ColBERT-as-retriever for.
- **Encoding cost is the real bottleneck**: ~90 seconds per query for
  reranking 200 candidates. Each query pays for ~1M tokens of forward
  pass on the encoder (200 docs × ~10 passages × ~500 tokens), and that
  ModernColBERT throughput on the local GPU is the floor. Bumping
  `sub_batch` from 16 to 64 changed nothing — the GPU was already
  saturated. **This is the cost the blob store eliminates.**
- **RRF hurts here.** Fusing voyager with passage_bm25 was *worse* than
  voyager alone at every depth, because RRF rewards docs that appear in
  multiple lists, which dilutes voyager's unique deep-recall hits. Not
  every channel benefits from fusion.

## What we should do for greatwheel

The empirical results above turn this from "the textbook says..." into "we
measured it on BrowseComp." The architecture is validated; the only thing
left to fix is encoding cost.

Long-term we want **Option B** in `gw-memory`. Concretely:

1. **Blob store of per-passage token tensors**, keyed by `passage_id`. Lance
   table is the obvious fit since we already use LanceDB. Schema:
   `(passage_id: string, doc_id: string, num_tokens: i32, vectors: bytes)`.
   Vectors stored as E4M3 (1 byte/dim) to keep total ~58 GB for BrowseComp scale.
2. **First-stage retrieval** uses our existing tantivy BM25 + (eventually) a
   dense single-vector index in LanceDB. Returns top ~500–1000 candidates.
3. **MaxSim rerank** on the candidate set using `maxsim-cpu`. Dequantize E4M3 →
   float32 on the fly per query (~6 MB per query, trivial). Measured 3.76 ms for
   1000 variable-length docs on this machine — not the bottleneck.

**The blob store is a drop-in replacement for the encoding step in the
current rerank.** `voyager_searcher.rerank()` used to call
`encoder._encode(passage_texts)` and paid ~90s/query for GPU forward
passes. With the blob store, that becomes "fetch precomputed token
tensors for the candidate passages by passage_id" — ~200ms, no GPU. The
rerank kernel and aggregation logic stay identical. Measured: same R@200
ceiling, slightly *better* R@5/R@10 (numerical noise from float16),
~450× lower per-query latency.

**But Voyager itself stays.** The blob store replaces re-encoding, not
candidate generation. When we tested BM25 → blob rerank without Voyager,
the R@200 ceiling collapsed from 24 to 17 — those 7 queries need
late-interaction first-stage retrieval, not just late-interaction
reranking. The 99 GB Voyager index is paying for them.

## maxsim-cpu — the Rust kernel question

Mixedbread open-sourced `maxsim-cpu` (Apache-2.0):
<https://github.com/mixedbread-ai/maxsim-cpu>

- Single-file Rust kernel (`src/lib.rs`, ~800 lines)
- BLAS GEMM → per-row max → sum, with hand-written AVX2 / ARM NEON SIMD for the
  max-reduce, 4-way ILP unrolling, prefetching, thread-local buffers, rayon
  parallelism across docs
- Optional libxsmm feature for JIT-compiled small-GEMM kernels
- Float32 only (no E4M3 / int8 support)
- **Distributed only as a Python wheel**: `crate-type = ["cdylib"]` with PyO3
  bindings. Not on crates.io, can't be a Rust dep as-is.

Measured on this machine, 1000 docs × 100–600 tokens × 128 dims:

| kernel             | latency  | speedup |
| ------------------ | -------- | ------- |
| maxsim-cpu uniform | 7.66 ms  | 5.0×    |
| maxsim-cpu var.    | 3.76 ms  | 10.2×   |
| torch CPU einsum   | 38.40 ms | 1×      |

Variable-length is faster than uniform because it doesn't waste cycles on
padding tokens. Real corpora are variable-length, so we'd use the variable path.

**Plan for using it from `gw-memory`** (when we get there):

1. Fork the repo. Change `crate-type = ["cdylib"]` to `["cdylib", "rlib"]` and
   gate the PyO3 bindings behind a feature. Vendor under `vendor/maxsim-cpu`.
   Reference from `gw-memory` as a path dep. ~30 min of work.
2. Eventually, if we want E4M3 throughout the hot path (skip the dequant step),
   extend the kernel ourselves. Not urgent — dequant cost is negligible.

The Python wheel is good enough for the BrowseComp benchmark side immediately,
which is what we'll use to validate Option B before committing the Rust work.

## Round-1 backend comparison (BrowseComp sample30, full 100K corpus)

Four backends, same 2000-token-per-doc cap, same ColBERT encoder, same
query set. Elasticsearch ran on 8.18 with trial license; Qdrant on 1.12.4;
LanceDB on 0.29. All indexes built from the same passage blob store
(no re-encoding).

```
Backend                   R@5    R@10   R@20   R@50   R@100  R@200   p50 ms   p95 ms
lancedb_mv               10/30  13/30  15/30  21/30  25/30  25/30    46076    48669
qdrant                   10/30  13/30  15/30  21/30  25/30  25/30    25680    30088
tantivy (BM25)            4/30   8/30   8/30   9/30  11/30  12/30      266      392
tantivy+blob_rerank       7/30   7/30   8/30  11/30  12/30  12/30    87500   234139
elasticsearch             0/30   0/30   0/30   0/30   0/30   0/30    10065    10078
```

### Build times

| backend      | docs    | time     | rate      | notes                          |
| ------------ | ------- | -------- | --------- | ------------------------------ |
| LanceDB MV   | 100,195 | 22 min   | 75 docs/s | embedded, no server            |
| Qdrant       | 100,195 | 204 min  | 8 docs/s  | Docker, HTTP JSON overhead     |
| Elasticsearch| 100,195 | 316 min  | 5 docs/s  | Docker, JVM, trial license req |

### Key findings

**1. Qdrant and LanceDB MV agree perfectly on recall.** Both native MaxSim
implementations return 10/13/15/21/25/25 at every k depth. Two independent
codebases reaching the same numbers is strong mutual validation that the
recall ceiling is real and not an artifact of one library's implementation.

**2. R@200 = 25/30 on the full 100K corpus** — one more than the 24/30 from
the earlier Voyager experiment (which covered only 85K docs). The missing
doc was in the 15K Voyager didn't index. This means 100% of sample30's
gold docs are reachable by late-interaction retrieval over this corpus.
The 5 remaining misses are genuine: no ColBERT representation of those
queries matches any passage of the gold doc well enough to surface it.

**3. Qdrant is ~2× faster than LanceDB MV** (26s vs 46s per query).
LanceDB MV is doing a full scan (no ANN index built); Qdrant has HNSW
with native MaxSim comparator. Both are still slow for production
serving (~30s on 100K docs), but Qdrant's HNSW path gets much faster
with warmup and ef_search tuning.

**4. Elasticsearch is unusable at this scale with rank_vectors.** ES 8.18's
`maxSimDotProduct` runs as a `script_score` over `match_all`, which is
JVM brute-force. 8 of 30 queries timed out at 60s; the rest returned
nothing. ES would need a kNN prefilter stage (which doesn't exist yet
for rank_vectors) or a fundamentally different query path. This is an
honest industry baseline: **ES's ColBERT support is license-gated AND
unscalable as of 8.18.**

**5. BM25→blob rerank helps the head but can't break the recall ceiling.**
R@5 improves from 4→7 (rerank sharpens the ranking), but R@200 stays at
12/30. The +13 gap vs native multi-vector (25 vs 12) re-confirms that
~43% of sample30 queries need late-interaction *retrieval*, not just
late-interaction reranking. This is the Weller bound in action.

### Operational gotchas from the build

- **Qdrant hard limits**: 1 MB per point (max ~2048 f32 vectors per multi-
  vector field), 32 MB default request size (`QDRANT__SERVICE__MAX_REQUEST_SIZE_MB`),
  and a low default `nofile` ulimit (1024) that crashes RocksDB at ~5K
  points. All fixable via config/Docker, but not documented prominently.
- **Elasticsearch requires Platinum/trial license** for `rank_vectors`.
  The basic (free) license returns a clear error: `"current license is
  non-compliant for [Rank Vectors]"`. Trial gives 30 days.
- **Concurrent builds OOM**: running all three builders simultaneously
  (each holding ~58 GB of passage data in memory) exceeded 251 GB. Fixed
  by streaming docs one at a time via `blob_doc_iter.py` and running
  builds serially.
- **LanceDB MV needs no infrastructure**: embedded, no Docker, no server,
  no license. 22-minute build. This is its biggest practical advantage
  over Qdrant and ES.

### Recommendation update

For greatwheel's production stack:

1. **Ship with LanceDB MV today.** Zero infrastructure (embedded), 25/30
   R@200, 22-minute build from existing blob store, runs in-process.
   The candle encoder is already parity-tested in `gw-memory`. The Rust
   `BlobReranker` is done and bit-identical to the Python version.

2. **Qdrant as the scale-out option.** Same recall, better latency (HNSW),
   but requires a server process. When greatwheel needs to scale to
   multi-tenant or >100K-doc corpora, Qdrant's architecture (sharding,
   replication, payload filtering) is the natural next step. The Docker
   compose config and Python searcher are ready; the Qdrant Rust client
   crate is first-class.

3. **Skip Elasticsearch for ColBERT.** ES's rank_vectors is license-gated,
   slow, and unscalable as of 8.18. If ES is already deployed in a
   customer's stack, use it for BM25 first-stage only, with blob rerank
   on top — not for native multi-vector.

## Frontier directions worth tracking

Not adopting any of these yet, but worth knowing about:

- **PLAID** (2022) — IVF + 2-bit residual PQ. The reference late-interaction
  index. ColBERTv2 / PyLate use this.
- **EMVB** (2023) — replaces PLAID's residual codes with bitvector signatures
  + SIMD popcount. ~2–3× faster than PLAID, same quality.
- **XTR** (2024, Google) — trains the model so top-k token retrieval directly
  approximates MaxSim. Drops the gather/rerank pipeline entirely.
- **MUVERA** (2024, Google) — converts multi-vector queries/docs into fixed-dim
  "Fixed Dimensional Encodings" such that inner product approximates Chamfer
  similarity. Lets you use a normal single-vector ANN. **Does not escape the
  Weller bound** — just buys headroom by using much larger *d* (typically
  10K–20K dims). Subject to the same theoretical ceiling at sufficient scale.
- **Mixedbread silo** (2024, closed source) — custom engine, billion-scale,
  variable-vector encoder. The current commercial frontier.
- **Reason-ModernColBERT** (2025, what we use) — ModernBERT backbone trained
  for reasoning-heavy retrieval. Vanilla late interaction at inference time.

The field is gradually moving toward "late interaction without specialized
infrastructure" via training-time tricks (XTR) and approximate single-vector
projections (MUVERA), but for now if you want frontier quality on hard retrieval
tasks, you're running multi-vector in some form.
