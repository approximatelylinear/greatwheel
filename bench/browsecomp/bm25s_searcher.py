"""
BM25S searcher for BrowseComp-Plus — pure Python, no Java needed.

Implements the BaseSearcher interface using bm25s for sparse retrieval.

First-time usage builds the index from the HuggingFace corpus:
    python bm25s_searcher.py --build-index --index-path ./data/bm25s-index

Then use as a searcher with ollama_client.py:
    python ollama_client.py --searcher-type custom ...
    (or import BM25sSearcher directly)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import bm25s

VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "BrowseComp-Plus"

# Import BaseSearcher directly to avoid triggering pyserini/faiss imports
# from the searchers __init__.py
import importlib.util
_base_spec = importlib.util.spec_from_file_location(
    "searchers.base",
    VENDOR_ROOT / "searcher" / "searchers" / "base.py",
)
_base_mod = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base_mod)
BaseSearcher = _base_mod.BaseSearcher

logger = logging.getLogger(__name__)


class BM25sSearcher(BaseSearcher):
    """BrowseComp-Plus searcher backed by bm25s (pure Python BM25)."""

    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--index-path",
            required=True,
            help="Path to bm25s index directory (created by --build-index)",
        )

    def __init__(self, args):
        self.index_path = args.index_path

        logger.info(f"Loading bm25s index from {self.index_path}")
        self.retriever = bm25s.BM25.load(self.index_path, mmap=True)

        # Load corpus texts + docids alongside the index
        import json
        meta_path = Path(self.index_path) / "corpus_meta.jsonl"
        self.docids: list[str] = []
        self.texts: dict[str, str] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.docids.append(obj["docid"])
                self.texts[obj["docid"]] = obj["text"]

        logger.info(f"bm25s index loaded: {len(self.docids)} documents")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self.retriever.retrieve(query_tokens, k=k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.docids):
                continue
            docid = self.docids[idx]
            results.append({
                "docid": docid,
                "score": float(score),
                "text": self.texts[docid],
            })

        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        text = self.texts.get(docid)
        if text is None:
            return None
        return {"docid": docid, "text": text}

    @property
    def search_type(self) -> str:
        return "BM25s"


def build_index(args):
    """Build bm25s index from the BrowseComp-Plus corpus."""
    import json
    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading BrowseComp-Plus corpus from HuggingFace...")
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    print(f"Loaded {len(ds)} documents")

    corpus_texts = ds["text"]
    corpus_docids = ds["docid"]

    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=True)

    print("Building BM25 index...")
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=True)

    # Save index
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_path))

    # Save corpus metadata (docids + texts) alongside the index
    meta_path = index_path / "corpus_meta.jsonl"
    print(f"Saving corpus metadata to {meta_path}...")
    with open(meta_path, "w", encoding="utf-8") as f:
        for docid, text in tqdm(zip(corpus_docids, corpus_texts), total=len(corpus_docids)):
            json.dump({"docid": docid, "text": text}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Index built at {index_path} ({len(corpus_docids)} documents)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bm25s index builder / searcher")
    parser.add_argument("--build-index", action="store_true", help="Build the index")
    parser.add_argument(
        "--index-path",
        default="./data/bm25s-index",
        help="Index directory path",
    )
    args = parser.parse_args()

    if args.build_index:
        build_index(args)
    else:
        print("Use --build-index to create the index, or import BM25sSearcher.")
