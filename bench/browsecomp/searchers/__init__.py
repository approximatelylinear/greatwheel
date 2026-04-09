"""Searcher backends for the BrowseComp retrieval benchmark.

Each backend implements the `Searcher` protocol from `base.py` and is
selectable via `retrieval_benchmark.py --searcher <name>`. The shared
encoder service (`colbert_server.py`) loads Reason-ModernColBERT once and
serves token vectors to whichever backend needs them.

See `docs/design-colbert-production.md` for the design rationale and
the empirical comparisons.
"""
