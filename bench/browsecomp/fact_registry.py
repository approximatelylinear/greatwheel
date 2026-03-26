"""
FactRegistry — Structured evidence accumulator for BrowseComp research sessions.

Inspired by Hindsight's Retain/Reflect operations (arXiv:2512.12818).
Replaces ad-hoc variable accumulation with typed, entity-tagged facts
and confidence-scored candidate answers.

This file is the SINGLE SOURCE OF TRUTH for the FactRegistry class.
It is used in two ways:
  1. Imported directly by bench/browsecomp/ollama_client.py
  2. Embedded at compile time into gw-bench via include_str! and executed
     inside ouros REPL sessions

For this reason, the code avoids dataclasses, type annotations on function
signatures, and from __future__ imports — these are not reliably supported
when run through exec().

Usage in REPL:
    facts.add("Marie Curie won the 1903 Nobel Prize in Physics", source="doc_42")
    facts.propose("Marie Curie", confidence=0.7, evidence="doc_42 mentions award")
    facts.reinforce("Marie Curie", evidence="doc_89 confirms")
    facts.summary()          # grouped view of all facts
    facts.candidates()       # ranked candidate answers
    facts.best_candidate()   # highest-confidence answer
"""

import re as _re

# Confidence adjustment constants (from Hindsight CARA)
_REINFORCE_ALPHA = 0.15
_WEAKEN_ALPHA = 0.10
_CONTRADICT_ALPHA = 0.25


def _extract_entities(text):
    """Extract entity mentions from text.

    Simple heuristic: capitalized multi-word phrases, quoted strings, numbers,
    and date-like patterns.  Good enough for entity linking within a session.
    """
    entities = []

    # Quoted strings
    for m in _re.finditer(r'"([^"]{2,60})"', text):
        entities.append(m.group(1).strip())

    # Capitalized phrases (2+ words starting with uppercase)
    for m in _re.finditer(r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|de|von|van|le|la|del|al))+)\b', text):
        phrase = m.group(1).strip()
        if len(phrase) > 3 and phrase not in ("The", "This", "That", "These", "Those"):
            entities.append(phrase)

    # Single capitalized words that look like proper nouns (not sentence starters)
    for m in _re.finditer(r'(?<=[a-z,;:]\s)([A-Z][a-z]{2,})\b', text):
        word = m.group(1)
        if word not in ("The", "This", "That", "However", "Also", "Additionally",
                        "Furthermore", "Moreover", "Therefore", "Because", "Although"):
            entities.append(word)

    # Years (4-digit numbers that look like years)
    for m in _re.finditer(r'\b(1[5-9]\d{2}|20\d{2})\b', text):
        entities.append(m.group(1))

    # Deduplicate preserving order
    seen = set()
    unique = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _normalize_answer(a):
    """Normalize a candidate answer for matching."""
    return _re.sub(r'\s+', ' ', a.strip().lower())


class FactRegistry:
    """Structured evidence accumulator for research sessions.

    Maintains two collections:
    - facts: extracted information from documents, tagged with entities and provenance
    - candidates: proposed answers with confidence scores that update with evidence

    Designed to be injected into the agent's REPL namespace alongside
    search(), get_document(), and llm_query().
    """

    def __init__(self):
        self._facts = []           # list of dicts: {text, source, kind, entities, iteration}
        self._candidates = []      # list of dicts: {answer, confidence, evidence, iteration}
        self._entity_index = {}    # entity_lower -> list of fact indices
        self._iteration = 0

    def set_iteration(self, n):
        """Called by the harness to track which iteration we're on."""
        self._iteration = n

    # ----- Fact management ----- #

    def add(self, text, source="", kind="fact"):
        """Add extracted fact(s). Multi-line text is split into individual facts.

        Returns a short confirmation string (for REPL output).
        """
        lines = [ln.strip().lstrip("*-\u2022").strip()
                 for ln in str(text).strip().split("\n") if ln.strip()]

        added = 0
        for line in lines:
            if len(line) < 5:
                continue
            entities = _extract_entities(line)
            idx = len(self._facts)
            self._facts.append({
                "text": line,
                "source": source,
                "kind": kind,
                "entities": entities,
                "iteration": self._iteration,
            })
            for ent in entities:
                self._entity_index.setdefault(ent.lower(), []).append(idx)
            added += 1

        return f"Added {added} fact(s) from {source or 'unknown'}. Total: {len(self._facts)} facts."

    def for_entity(self, entity):
        """All facts mentioning a specific entity."""
        indices = self._entity_index.get(entity.lower(), [])
        return [f"[{self._facts[i]['source']}] {self._facts[i]['text']}" for i in indices]

    def entities(self):
        """All known entities, sorted by frequency (most mentioned first)."""
        counts = {}
        canonical = {}
        for f in self._facts:
            for ent in f["entities"]:
                key = ent.lower()
                counts[key] = counts.get(key, 0) + 1
                if key not in canonical:
                    canonical[key] = ent
        return [canonical[k] for k, _ in sorted(counts.items(), key=lambda x: -x[1])]

    # ----- Candidate management ----- #

    def propose(self, answer, confidence=0.5, evidence=""):
        """Propose a candidate answer with confidence score.

        If the candidate already exists, updates confidence to the max.
        """
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c["answer"]) == norm:
                c["confidence"] = max(c["confidence"], confidence)
                if evidence:
                    c["evidence"].append(evidence)
                return f"Updated '{c['answer']}' -> confidence {c['confidence']:.2f} ({len(c['evidence'])} evidence)"
        cand = {
            "answer": answer.strip(),
            "confidence": min(max(confidence, 0.0), 1.0),
            "evidence": [evidence] if evidence else [],
            "iteration": self._iteration,
        }
        self._candidates.append(cand)
        return f"Proposed '{cand['answer']}' with confidence {cand['confidence']:.2f}"

    def reinforce(self, answer, evidence=""):
        """Increase confidence for a candidate (+0.15)."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c["answer"]) == norm:
                c["confidence"] = min(c["confidence"] + _REINFORCE_ALPHA, 1.0)
                if evidence:
                    c["evidence"].append(evidence)
                return f"Reinforced '{c['answer']}' -> {c['confidence']:.2f}"
        return self.propose(answer, confidence=0.5 + _REINFORCE_ALPHA, evidence=evidence)

    def contradict(self, answer, reason=""):
        """Decrease confidence for a candidate (-0.25)."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c["answer"]) == norm:
                c["confidence"] = max(c["confidence"] - _CONTRADICT_ALPHA, 0.0)
                if reason:
                    c["evidence"].append(f"[CONTRADICTED] {reason}")
                return f"Contradicted '{c['answer']}' -> {c['confidence']:.2f}"
        return f"No candidate '{answer}' found to contradict."

    def weaken(self, answer, reason=""):
        """Slightly decrease confidence for a candidate (-0.10)."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c["answer"]) == norm:
                c["confidence"] = max(c["confidence"] - _WEAKEN_ALPHA, 0.0)
                if reason:
                    c["evidence"].append(f"[WEAKENED] {reason}")
                return f"Weakened '{c['answer']}' -> {c['confidence']:.2f}"
        return f"No candidate '{answer}' found to weaken."

    def candidates(self):
        """Ranked candidate answers with confidence and evidence count."""
        if not self._candidates:
            return "No candidates proposed yet."
        sc = sorted(self._candidates, key=lambda c: -c["confidence"])
        lines = ["Candidate answers (ranked by confidence):"]
        for i, c in enumerate(sc):
            m = "->" if i == 0 else "  "
            lines.append(
                f"  {m} {c['answer']} (confidence: {c['confidence']:.2f}, "
                f"evidence: {len(c['evidence'])} items)"
            )
        return "\n".join(lines)

    def best_candidate(self):
        """Highest-confidence candidate answer, or None."""
        if not self._candidates:
            return None
        best = max(self._candidates, key=lambda c: c["confidence"])
        return (best["answer"], best["confidence"])

    # ----- Summaries ----- #

    def summary(self):
        """Deduplicated, entity-grouped fact summary."""
        if not self._facts:
            return "No facts collected yet."

        lines = [f"=== Fact Summary ({len(self._facts)} facts, "
                 f"{len(self._entity_index)} entities) ===\n"]

        # Group by entity (show top entities by frequency)
        top_ents = self.entities()[:10]
        shown = set()

        for ent in top_ents:
            indices = self._entity_index.get(ent.lower(), [])
            if not indices:
                continue
            lines.append(f"[{ent}]")
            for idx in indices[:5]:
                f = self._facts[idx]
                lines.append(f"  * {f['text']} (from {f['source']})")
                shown.add(idx)
            if len(indices) > 5:
                lines.append(f"  ... and {len(indices) - 5} more")
            lines.append("")

        # Show ungrouped facts
        ungrouped = [i for i in range(len(self._facts)) if i not in shown]
        if ungrouped:
            lines.append("[Other facts]")
            for idx in ungrouped[:10]:
                f = self._facts[idx]
                lines.append(f"  * {f['text']} (from {f['source']})")
            if len(ungrouped) > 10:
                lines.append(f"  ... and {len(ungrouped) - 10} more")

        # Append candidates
        if self._candidates:
            lines.append("")
            lines.append(self.candidates())

        return "\n".join(lines)

    def __repr__(self):
        n = len(self._facts)
        nc = len(self._candidates)
        ne = len(self._entity_index)
        best = self.best_candidate()
        bs = f", best='{best[0]}' ({best[1]:.2f})" if best else ""
        return f"FactRegistry({n} facts, {ne} entities, {nc} candidates{bs})"


# When executed via exec() in ouros, auto-instantiate `facts` in the namespace.
# When imported as a module, this is harmless (caller creates their own instance).
facts = FactRegistry()
