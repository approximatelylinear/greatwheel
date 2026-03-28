"""ColBERT reranker for BrowseComp-Plus search — scores query-document pairs via MaxSim.

Loads a ColBERT model (default: Reason-ModernColBERT) and reranks BM25 candidate
documents. No multi-vector index needed — documents are encoded on-the-fly.

Usage (standalone test):
    python colbert_reranker.py --query "Who won the Nobel Prize in Physics 2024?"
"""

import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


class ColBERTReranker:
    """Rerank documents using ColBERT MaxSim scoring."""

    def __init__(self, model_name: str = "lightonai/Reason-ModernColBERT", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name

    @torch.no_grad()
    def _encode(self, texts: list[str], is_query: bool, max_length: int = 8192) -> torch.Tensor:
        """Encode texts into token-level embeddings (batch × tokens × dim)."""
        if is_query:
            max_length = 128  # ColBERT query limit
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**encoded)
        # Use last hidden state as token embeddings
        return outputs.last_hidden_state  # (batch, seq_len, dim)

    @torch.no_grad()
    def _maxsim(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> list[float]:
        """Compute MaxSim scores: for each query token, find max similarity to any doc token."""
        # query_emb: (1, q_tokens, dim)
        # doc_embs: (n_docs, d_tokens, dim)
        # Normalize
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        doc_embs = torch.nn.functional.normalize(doc_embs, dim=-1)

        scores = []
        q = query_emb.squeeze(0)  # (q_tokens, dim)
        for i in range(doc_embs.size(0)):
            d = doc_embs[i]  # (d_tokens, dim)
            # Similarity: (q_tokens, d_tokens)
            sim = torch.matmul(q, d.t())
            # MaxSim: for each query token, take max over doc tokens, then sum
            max_sim = sim.max(dim=1).values.sum().item()
            scores.append(max_sim)
        return scores

    def _condense_query(self, query: str, max_tokens: int = 120) -> str:
        """Condense long queries to fit ColBERT's 128-token limit.

        Strategy: keep the last sentence (usually the actual question)
        plus the most distinctive sentences from the rest. This preserves
        the question intent while fitting within the token budget.
        """
        tokens = self.tokenizer.encode(query)
        if len(tokens) <= max_tokens:
            return query  # Already fits

        # Split into sentences, keep the last one (the question)
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', query) if s.strip()]
        if not sentences:
            return self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

        # Last sentence is usually the question — always keep it
        question = sentences[-1]
        rest = sentences[:-1]

        # Score remaining sentences by distinctiveness (proportion of
        # capitalized words, numbers, and rare terms)
        def score_sentence(s: str) -> float:
            words = s.split()
            if not words:
                return 0
            caps = sum(1 for w in words if w[0].isupper() and len(w) > 1)
            nums = sum(1 for w in words if any(c.isdigit() for c in w))
            return (caps + nums * 2) / len(words)

        scored = [(score_sentence(s), s) for s in rest]
        scored.sort(key=lambda x: -x[0])

        # Build condensed query: question + most distinctive sentences
        result = question
        for _, sent in scored:
            candidate = sent + ". " + result
            if len(self.tokenizer.encode(candidate)) <= max_tokens:
                result = candidate
            else:
                break

        return result

    def rerank(self, query: str, documents: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
        """Rerank documents by ColBERT MaxSim score.

        Args:
            query: Search query string.
            documents: List of dicts with at least "docid" and "text" keys.
            k: Number of top results to return.

        Returns:
            Top-k documents sorted by ColBERT score descending.
        """
        if not documents:
            return []

        # Use full doc text — truncation to 8192 tokens happens in _encode
        doc_texts = [d.get("text", "") for d in documents]

        t0 = time.monotonic()
        # Condense long queries to fit ColBERT's 128-token limit
        condensed = self._condense_query(query)
        query_emb = self._encode([condensed], is_query=True)

        # Encode documents in batches to manage memory
        batch_size = 8
        all_scores: list[float] = []
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i + batch_size]
            doc_emb = self._encode(batch, is_query=False)
            batch_scores = self._maxsim(query_emb, doc_emb)
            all_scores.extend(batch_scores)

        elapsed_ms = (time.monotonic() - t0) * 1000

        scored = list(zip(all_scores, documents))
        scored.sort(key=lambda x: -x[0])

        results = []
        for score, doc in scored[:k]:
            results.append({**doc, "score": float(score)})

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ColBERT reranker smoke test")
    parser.add_argument("--query", default="Nobel Prize Physics 2024")
    parser.add_argument("--model", default="lightonai/Reason-ModernColBERT")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    reranker = ColBERTReranker(args.model)

    # Fake documents for testing
    docs = [
        {"docid": "1", "text": "The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton."},
        {"docid": "2", "text": "The weather in Stockholm was cold in December 2024."},
        {"docid": "3", "text": "Geoffrey Hinton is known as the godfather of deep learning."},
    ]

    results = reranker.rerank(args.query, docs, k=3)
    for r in results:
        print(f"  {r['docid']} score={r['score']:.4f}: {r['text'][:80]}")
