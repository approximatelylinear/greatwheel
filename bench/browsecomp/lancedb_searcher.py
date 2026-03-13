#!/usr/bin/env python3
"""
LanceDB-based searcher for BrowseComp-Plus.

This implements the BrowseComp-Plus BaseSearcher interface using LanceDB for
vector search and Ollama for embeddings — mirroring gw-memory's planned architecture.

Usage:
    # Build the index first
    python lancedb_searcher.py --build-index \
        --ollama-url http://localhost:11434 \
        --embedding-model nomic-embed-text \
        --db-path ./data/lancedb

    # Then use as a searcher with the MCP server or ollama_client.py
    # (registered as 'custom' searcher type, or import directly)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import pyarrow as pa
import requests
from tqdm import tqdm

# Import BaseSearcher directly to avoid triggering pyserini/faiss imports
VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "BrowseComp-Plus"

import importlib.util
_base_spec = importlib.util.spec_from_file_location(
    "searchers.base",
    VENDOR_ROOT / "searcher" / "searchers" / "base.py",
)
_base_mod = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base_mod)
BaseSearcher = _base_mod.BaseSearcher

logger = logging.getLogger(__name__)


def embed_texts(
    texts: list[str],
    ollama_url: str,
    model: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts using Ollama's embedding API."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = requests.post(
                f"{ollama_url}/api/embed",
                json={"model": model, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(data["embeddings"])
        except Exception:
            # Retry individually — some texts may be too long or malformed
            for text in batch:
                try:
                    resp = requests.post(
                        f"{ollama_url}/api/embed",
                        json={"model": model, "input": [text[:4096]]},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    all_embeddings.extend(resp.json()["embeddings"])
                except Exception:
                    # Use zero vector as fallback
                    if all_embeddings:
                        dim = len(all_embeddings[0])
                    else:
                        dim = 768
                    all_embeddings.append([0.0] * dim)

    return np.array(all_embeddings, dtype=np.float32)


class LanceDBSearcher(BaseSearcher):
    """BrowseComp-Plus searcher backed by LanceDB + Ollama embeddings."""

    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--db-path",
            default="./data/lancedb",
            help="Path to LanceDB database directory",
        )
        parser.add_argument(
            "--table-name",
            default="browsecomp_docs",
            help="LanceDB table name (default: browsecomp_docs)",
        )
        parser.add_argument(
            "--ollama-url",
            default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            help="Ollama API URL for embeddings",
        )
        parser.add_argument(
            "--embedding-model",
            default=os.getenv("GW_EMBED_MODEL", "nomic-embed-text"),
            help="Ollama embedding model (default: nomic-embed-text)",
        )

    def __init__(self, args):
        self.db_path = args.db_path
        self.table_name = getattr(args, "table_name", "browsecomp_docs")
        self.ollama_url = getattr(args, "ollama_url", "http://localhost:11434")
        self.embedding_model = getattr(args, "embedding_model", "nomic-embed-text")

        logger.info(f"Opening LanceDB at {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        self.table = self.db.open_table(self.table_name)
        logger.info(f"Opened table '{self.table_name}' with {self.table.count_rows()} rows")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return embed_texts([query], self.ollama_url, self.embedding_model)[0]

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        q_vec = self._embed_query(query)
        results = (
            self.table.search(q_vec.tolist())
            .limit(k)
            .to_list()
        )

        return [
            {
                "docid": str(r["docid"]),
                "score": float(r.get("_distance", 0.0)),
                "text": r["text"],
            }
            for r in results
        ]

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        results = self.table.search().where(f"docid = '{docid}'").limit(1).to_list()
        if not results:
            return None
        r = results[0]
        return {"docid": str(r["docid"]), "text": r["text"]}

    @property
    def search_type(self) -> str:
        return "LanceDB"

    def search_description(self, k: int = 10) -> str:
        return (
            f"Search a LanceDB vector index of ~100K web documents. "
            f"Returns top-{k} hits by semantic similarity with docid, score, and snippet."
        )


# --------------------------------------------------------------------------- #
# Index builder
# --------------------------------------------------------------------------- #

def build_index(args):
    """Build the LanceDB index from the BrowseComp-Plus corpus.

    Supports --resume to continue from an interrupted build by skipping
    documents that are already indexed.
    """
    from datasets import load_dataset

    print("Loading BrowseComp-Plus corpus from HuggingFace...")
    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    print(f"Loaded {len(ds)} documents")

    db = lancedb.connect(args.db_path)

    # Check embedding dimension
    test_emb = embed_texts(["test"], args.ollama_url, args.embedding_model)
    dim = test_emb.shape[1]
    print(f"Embedding dimension: {dim} (model: {args.embedding_model})")

    batch_size = args.embed_batch_size
    total = len(ds)

    schema = pa.schema([
        pa.field("docid", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    # Resume support: find already-indexed docids
    existing_docids = set()
    table = None
    if args.resume:
        try:
            table = db.open_table(args.table_name)
            existing_count = table.count_rows()
            # Scan all docids already in the table
            existing_docids = set(
                r["docid"] for r in table.search().select(["docid"]).limit(existing_count + 1000).to_list()
            )
            print(f"Resuming: {len(existing_docids)} documents already indexed")
        except Exception as e:
            print(f"No existing table found, starting fresh: {e}")

    for start in tqdm(range(0, total, batch_size), desc="Embedding & indexing"):
        end = min(start + batch_size, total)
        batch = ds[start:end]

        docids = batch["docid"]
        texts = batch["text"]

        # Skip already-indexed documents
        if existing_docids:
            keep = [i for i, d in enumerate(docids) if d not in existing_docids]
            if not keep:
                continue
            docids = [docids[i] for i in keep]
            texts = [texts[i] for i in keep]

        truncated = [t[:8192] for t in texts]
        embeddings = embed_texts(truncated, args.ollama_url, args.embedding_model)

        data = pa.table(
            {
                "docid": docids,
                "text": texts,
                "vector": embeddings.tolist(),
            },
            schema=schema,
        )

        if table is None:
            table = db.create_table(args.table_name, data, mode="overwrite")
        else:
            table.add(data)

    print(f"Index built: {table.count_rows()} documents in {args.db_path}/{args.table_name}")

    # Create IVF-PQ index for faster search
    print("Creating IVF-PQ index...")
    table.create_index(
        metric="cosine",
        num_partitions=256,
        num_sub_vectors=96,
    )
    print("Index created successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB searcher / index builder")
    parser.add_argument("--build-index", action="store_true", help="Build the index")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted index build")
    parser.add_argument("--db-path", default="./data/lancedb", help="LanceDB path")
    parser.add_argument("--table-name", default="browsecomp_docs", help="Table name")
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("GW_EMBED_MODEL", "nomic-embed-text"),
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=32, help="Batch size for embedding"
    )

    args = parser.parse_args()

    if args.build_index:
        build_index(args)
    else:
        print("Use --build-index to create the index, or import LanceDBSearcher from another script.")
