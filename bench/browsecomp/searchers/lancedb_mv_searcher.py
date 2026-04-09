"""LanceDB native multi-vector searcher.

Uses LanceDB 0.29+'s native MaxSim support: each row in the table holds
all of a doc's passage tokens flattened into a single multi-vector list,
and `tbl.search(query_token_matrix)` natively computes
sum-of-max-of-cosine-similarity (i.e. ColBERT MaxSim) in the C++ layer.

The trick that makes per-doc storage work: max-pooling is associative.
`max(max(p1_tokens), max(p2_tokens)) == max((p1 ∪ p2)_tokens)`. So
flattening all passage tokens of a doc into one list and then asking for
MaxSim against that list is exactly the per-doc-max-of-per-passage-max
behavior we validated empirically.

Index built by `bench/browsecomp/build_lancedb_mv_index.py` from the
existing passage blob store (no re-encoding).
"""

from __future__ import annotations

import os
import sys

import lancedb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from searchers.base import EncoderClient, ScoredDoc

TABLE = "docs_mv"


class LanceDbMvSearcher:
    name = "lancedb_mv"

    def __init__(self, index_dir: str, encoder: EncoderClient):
        self.encoder = encoder
        print(f"Opening LanceDB MV index: {index_dir}", flush=True)
        db = lancedb.connect(index_dir)
        self.table = db.open_table(TABLE)
        n = self.table.count_rows()
        print(f"  {n} docs", flush=True)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        # Encode query → (Nq, 128)
        q = self.encoder.encode_query(query)

        # Native multi-vector search. Returns _distance where lower = better
        # and _distance ≈ Nq - MaxSim (default cosine flavor). We invert
        # for the score field so higher = better, matching other backends.
        df = (
            self.table.search(q)
            .select(["docid"])
            .limit(k)
            .to_pandas()
        )

        results: list[ScoredDoc] = []
        for _, row in df.iterrows():
            results.append(ScoredDoc(
                docid=str(row["docid"]),
                score=float(-row["_distance"]),
            ))
        return results
