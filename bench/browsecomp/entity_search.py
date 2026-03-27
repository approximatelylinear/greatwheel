"""
entity_search() — one-hop entity-bridged BM25 search.

Requires `search()` and `_extract_entities()` to be available in the
REPL namespace (provided by fact_registry.py bootstrap and the search
bridge respectively).

This file is the SINGLE SOURCE OF TRUTH for entity_search.
It is used in two ways:
  1. exec()'d in ollama_client.py after REPL namespace setup
  2. Embedded at compile time into gw-bench via include_str!

Usage in REPL:
    hits = entity_search("Marie Curie", hops=1, k=5)
    # Returns documents mentioning Marie Curie AND documents mentioning
    # entities that co-occur with Marie Curie in the top results.
"""


def entity_search(entity, hops=1, k=5):
    """Search for an entity and follow co-occurring entities one hop.

    1. BM25 search for the entity name
    2. Extract entities from the top result snippets
    3. For each co-occurring entity (up to `hops` iterations):
       - BM25 search for that entity
       - Collect results
    4. Return the deduplicated union, ordered by first-seen score

    Args:
        entity: Entity name to search for (e.g., "Marie Curie")
        hops: Number of entity-chain hops (default 1)
        k: Results per individual search (default 5)

    Returns:
        List of {docid, score, snippet} dicts, deduplicated by docid.
    """
    seen_docids = set()
    all_results = []

    def _add_results(results):
        for r in results:
            did = r.get("docid", "")
            if did and did not in seen_docids:
                seen_docids.add(did)
                all_results.append(r)

    # Initial search for the entity
    initial = search(entity, k=k)
    _add_results(initial)

    # Extract co-occurring entities from snippets
    frontier_entities = set()
    for r in initial:
        snippet = r.get("snippet", "")
        if snippet:
            for ent in _extract_entities(snippet):
                # Skip the original entity and very short matches
                if ent.lower() != entity.lower() and len(ent) > 2:
                    frontier_entities.add(ent)

    # Follow entity chains for the specified number of hops
    for _hop in range(hops):
        if not frontier_entities:
            break

        next_frontier = set()
        for co_entity in list(frontier_entities)[:8]:  # Cap to avoid explosion
            hits = search(co_entity, k=k)
            _add_results(hits)

            # Extract next-hop entities from new results
            for r in hits:
                snippet = r.get("snippet", "")
                if snippet:
                    for ent in _extract_entities(snippet):
                        if (ent.lower() != entity.lower()
                                and ent not in frontier_entities
                                and len(ent) > 2):
                            next_frontier.add(ent)

        frontier_entities = next_frontier

    return all_results
