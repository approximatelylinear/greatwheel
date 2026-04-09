"""Voyager-backed ColBERT searcher.

Loads a Voyager HNSW index of flattened ColBERT token vectors plus a
token→docid map, encodes queries with [Q]-prefix Reason-ModernColBERT, and
retrieves docs by approximate MaxSim aggregation:

    1. Encode query → ~30 query token vectors (128-dim, normalized)
    2. For each query token, ANN-query Voyager for top-N nearest token vectors
    3. For each query token, take the BEST similarity per docid (this is the
       max in MaxSim — any other token from the same doc is dominated)
    4. Sum the per-query-token maxes across the query → final doc score
    5. Return top-k docs by total score

This is approximate MaxSim because we only see tokens that appear in some query
token's top-N neighborhood. Tokens from a doc that aren't near any query token
contribute zero (which is fine — they wouldn't have been the max anyway).

Usage (standalone test):
    uv run --project bench/browsecomp --extra colbert \\
        python -u bench/browsecomp/voyager_searcher.py \\
        --index data/voyager-passages \\
        --corpus vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \\
        --query "Who won the Nobel Prize in Physics 2024?"
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import voyager

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder
from build_voyager_index import split as split_passages, title as extract_title


class VoyagerSearcher:
    """Approximate MaxSim retrieval over a flat Voyager token index."""

    def __init__(
        self,
        index_dir: str,
        corpus_path: Optional[str] = None,
        model_name: str = "lightonai/Reason-ModernColBERT",
        device: Optional[str] = None,
        ef_search: int = 100,
        blob_store_dir: Optional[str] = None,
    ):
        idx_path = os.path.join(index_dir, "index.voyager")
        map_path = os.path.join(index_dir, "token_to_docid.pkl")
        progress_path = os.path.join(index_dir, "progress.json")

        print(f"Loading Voyager index: {idx_path}", flush=True)
        t0 = time.monotonic()
        self.index = voyager.Index.load(idx_path)
        self.index.ef = ef_search
        print(f"  loaded in {time.monotonic()-t0:.1f}s, {self.index.num_elements} tokens", flush=True)

        print(f"Loading token→docid map: {map_path}", flush=True)
        t0 = time.monotonic()
        with open(map_path, "rb") as f:
            self.token_to_docid = pickle.load(f)
        print(f"  loaded in {time.monotonic()-t0:.1f}s, {len(self.token_to_docid)} entries", flush=True)

        with open(progress_path) as f:
            self.progress = json.load(f)
        print(f"  index covers {self.progress['doc_count']} docs / "
              f"{self.progress['passage_count']} passages", flush=True)

        # Corpus text for returning hits (optional — only needed if you want .text)
        self.texts: Dict[str, str] = {}
        if corpus_path:
            print(f"Loading corpus text: {corpus_path}", flush=True)
            t0 = time.monotonic()
            with open(corpus_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.texts[str(obj["docid"])] = obj["text"]
            print(f"  loaded {len(self.texts)} docs in {time.monotonic()-t0:.1f}s", flush=True)

        print(f"Loading encoder: {model_name}", flush=True)
        self.encoder = ColBERTEncoder(model_name, device=device)
        print(f"  device: {self.encoder.device}", flush=True)

        # Optional precomputed passage blob store (for fast rerank, no GPU)
        self.blob_table = None
        if blob_store_dir:
            import lancedb
            print(f"Loading blob store: {blob_store_dir}", flush=True)
            t0 = time.monotonic()
            blob_db = lancedb.connect(blob_store_dir)
            self.blob_table = blob_db.open_table("passage_blobs")
            n_rows = self.blob_table.count_rows()
            print(f"  loaded in {time.monotonic()-t0:.1f}s, {n_rows} passages", flush=True)

    def search(
        self,
        query: str,
        k: int = 10,
        tokens_per_query: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k docs by approximate MaxSim.

        Args:
            query: query string
            k: number of docs to return
            tokens_per_query: ANN top-N per query token. Higher = more recall,
                slower. 1000 is a reasonable default.
        """
        # 1. Encode query → (Nq, 128) normalized float32 numpy
        query_vecs = self.encoder._encode([query], max_length=128, is_query=True)[0].numpy()
        query_vecs = query_vecs.astype(np.float32)

        # 2. Batch ANN query: shape (Nq, tokens_per_query)
        # Voyager's query() accepts a batch of vectors directly.
        neighbors, distances = self.index.query(query_vecs, k=tokens_per_query)
        # cosine distance → cosine similarity
        similarities = 1.0 - distances

        # 3. Per-query-token, take best similarity per docid (the "max" in MaxSim).
        # 4. Sum across query tokens → per-doc score.
        doc_scores: Dict[str, float] = defaultdict(float)
        nq = query_vecs.shape[0]
        for qi in range(nq):
            best_per_doc: Dict[str, float] = {}
            row_neighbors = neighbors[qi]
            row_sims = similarities[qi]
            for tok_id, sim in zip(row_neighbors, row_sims):
                docid = self.token_to_docid.get(int(tok_id))
                if docid is None:
                    continue
                prev = best_per_doc.get(docid)
                if prev is None or sim > prev:
                    best_per_doc[docid] = float(sim)
            for docid, sim in best_per_doc.items():
                doc_scores[docid] += sim

        # 5. Top-k
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results = []
        for docid, score in ranked:
            results.append({
                "docid": docid,
                "score": score,
                "text": self.texts.get(docid, ""),
            })
        return results

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        candidate_texts: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates by full per-passage MaxSim using maxsim-cpu.

        For each candidate doc:
          1. Split into 4096-char passages with 800-char overlap (same chunking
             as the Voyager build, so we score the same units we indexed).
          2. Encode every passage on GPU with [D] prefix + title.
          3. MaxSim each passage against the query.
          4. Take MAX across passages → per-doc score.

        This avoids the "first-passage-only" trap of the previous version: the
        relevant content might live in passage #3 of a doc, and naive doc-level
        rerank with a 4096-char truncation misses it entirely.
        """
        import torch

        # Encode query (with [Q] prefix). Keep on CPU as a torch tensor — the
        # rerank kernel is negligible compared to GPU encoding, so we use plain
        # torch instead of maxsim-cpu (which segfaults in libxsmm under repeat use).
        q = self.encoder._encode([query], max_length=128, is_query=True)[0]  # (Nq, 128) torch CPU

        # Build flat list of (candidate_idx, passage_text), tracking ownership
        passage_owners: List[int] = []  # candidate_idx for each passage
        passage_texts: List[str] = []
        for ci, c in enumerate(candidates):
            text = candidate_texts.get(c["docid"], "")
            if not text:
                continue
            t = extract_title(text)
            pfx = f"[Title: {t}]\n" if t else ""
            for chunk in split_passages(text):
                passage_owners.append(ci)
                passage_texts.append(pfx + chunk)

        if not passage_texts:
            return candidates

        # Encode all passages in one go (sub-batched internally)
        passage_tensors = self.encoder._encode(passage_texts, max_length=512, is_query=False)

        # Real MaxSim via plain torch: for each passage, sum over query tokens
        # of (max over passage tokens of q · p). Both are L2-normalized so dot
        # product == cosine similarity.
        passage_scores = []
        for pt in passage_tensors:
            # pt: (Np, 128), q: (Nq, 128) → sims: (Nq, Np)
            sims = q @ pt.T
            passage_scores.append(float(sims.max(dim=1).values.sum()))

        # Per-doc max across passages
        doc_max: Dict[int, float] = {}
        for owner, s in zip(passage_owners, passage_scores):
            s = float(s)
            prev = doc_max.get(owner)
            if prev is None or s > prev:
                doc_max[owner] = s

        # Build reranked list. Docs we couldn't encode (no text) sink to -inf.
        scored = [
            {
                "docid": c["docid"],
                "score": doc_max.get(i, float("-inf")),
                "text": c.get("text", ""),
            }
            for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def rerank_from_blobs(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates by full per-passage MaxSim using precomputed blobs.

        Same semantics as `rerank()` but fetches token tensors from the
        Lance blob store instead of re-encoding on GPU. Encoding is the
        ~90s/query bottleneck; this should drop rerank latency to <1s.
        """
        if self.blob_table is None:
            raise RuntimeError("blob_store_dir was not provided at construction time")
        import torch

        # Encode just the query — cheap, ~few ms
        q = self.encoder._encode([query], max_length=128, is_query=True)[0]  # (Nq, 128) torch CPU

        docids = [c["docid"] for c in candidates]
        if not docids:
            return candidates

        # Fetch all passages for the candidate docs in one Lance query.
        # SQL-style IN filter — Lance handles this with a hash join.
        quoted = ",".join(f"'{d}'" for d in docids)
        arrow_tbl = (
            self.blob_table.search()
            .where(f"docid IN ({quoted})")
            .select(["docid", "chunk_idx", "num_tokens", "vectors"])
            .limit(len(docids) * 64)  # generous upper bound: 64 passages/doc
            .to_arrow()
        )

        # Group passages by docid, score each, take per-doc max
        doc_max: Dict[str, float] = {}
        col_docid = arrow_tbl.column("docid").to_pylist()
        col_ntok = arrow_tbl.column("num_tokens").to_pylist()
        col_vecs = arrow_tbl.column("vectors").to_pylist()

        for docid, n, vec_bytes in zip(col_docid, col_ntok, col_vecs):
            arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n, 128)
            pt = torch.from_numpy(arr.astype(np.float32))  # cast for matmul
            sims = q @ pt.T  # (Nq, n)
            score = float(sims.max(dim=1).values.sum())
            prev = doc_max.get(docid)
            if prev is None or score > prev:
                doc_max[docid] = score

        scored = [
            {
                "docid": c["docid"],
                "score": doc_max.get(c["docid"], float("-inf")),
                "text": c.get("text", ""),
            }
            for c in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        text = self.texts.get(docid)
        if text is None:
            return None
        return {"docid": docid, "text": text}

    @property
    def search_type(self) -> str:
        return "voyager-colbert"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/voyager-passages")
    parser.add_argument("--corpus", default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--tokens-per-query", type=int, default=1000)
    parser.add_argument("--ef", type=int, default=100)
    args = parser.parse_args()

    searcher = VoyagerSearcher(args.index, args.corpus, ef_search=args.ef)

    t0 = time.monotonic()
    results = searcher.search(args.query, k=args.k, tokens_per_query=args.tokens_per_query)
    dt = time.monotonic() - t0
    print(f"\nQuery: {args.query!r}")
    print(f"Latency: {dt*1000:.0f} ms\n")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:120].replace("\n", " ")
        print(f"  {i:2d}. [{r['score']:.3f}] {r['docid']}  {snippet}")


if __name__ == "__main__":
    main()
