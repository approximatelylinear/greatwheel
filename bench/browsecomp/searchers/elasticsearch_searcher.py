"""Elasticsearch rank_vectors searcher (industry baseline).

Uses ES 8.18+'s native `rank_vectors` field type and the `maxSimDotProduct`
script function to compute ColBERT MaxSim at query time. The query is the
encoded token matrix; ES returns docs ranked by sum-of-max-of-dot-product
across query tokens.

Index is built by `bench/browsecomp/build_elasticsearch_index.py` from
the existing passage blob store (one ES doc per corpus doc, all passage
tokens flattened).
"""

from __future__ import annotations

import os
import sys

from elasticsearch import Elasticsearch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from searchers.base import EncoderClient, ScoredDoc


class ElasticsearchSearcher:
    name = "elasticsearch"

    def __init__(
        self,
        encoder: EncoderClient,
        es_url: str = "http://localhost:9200",
        index: str = "colbert_mv",
    ):
        self.encoder = encoder
        self.es = Elasticsearch(es_url)
        self.index = index
        n = self.es.count(index=index).body["count"]
        print(f"ES index {index!r}: {n} docs", flush=True)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        # Encode query → list of 128-dim float lists for the script param.
        q_arr = self.encoder.encode_query(query)
        q_list = q_arr.tolist()

        # script_score over a match_all base — runs maxSimDotProduct on every doc.
        # For ~100K-doc corpora this is fine; for larger you'd add a kNN prefilter.
        body = {
            "size": k,
            "_source": ["docid"],
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "maxSimDotProduct(params.query_vector, 'tokens')",
                        "params": {"query_vector": q_list},
                    },
                }
            },
        }

        resp = self.es.search(index=self.index, body=body)
        hits = resp.body["hits"]["hits"]
        return [
            ScoredDoc(
                docid=h["_source"]["docid"],
                score=float(h["_score"]),
            )
            for h in hits
        ]
