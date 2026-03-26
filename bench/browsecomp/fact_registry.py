"""
FactRegistry — Structured evidence accumulator for BrowseComp research sessions.

Inspired by Hindsight's Retain/Reflect operations (arXiv:2512.12818).
Replaces ad-hoc variable accumulation with typed, entity-tagged facts
and confidence-scored candidate answers.

Usage in REPL:
    facts.add("Marie Curie won the 1903 Nobel Prize in Physics", source="doc_42")
    facts.propose("Marie Curie", confidence=0.7, evidence="doc_42 mentions award")
    facts.reinforce("Marie Curie", evidence="doc_89 confirms")
    facts.summary()          # grouped view of all facts
    facts.candidates()       # ranked candidate answers
    facts.best_candidate()   # highest-confidence answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Confidence adjustment constants (from Hindsight CARA)
REINFORCE_ALPHA = 0.15
WEAKEN_ALPHA = 0.10
CONTRADICT_ALPHA = 0.25


@dataclass
class Fact:
    """A single extracted fact with provenance."""
    text: str
    source: str  # docid
    kind: str = "fact"  # fact | observation | experience
    entities: list[str] = field(default_factory=list)
    iteration: int = -1


@dataclass
class Candidate:
    """A candidate answer with confidence tracking."""
    answer: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    iteration: int = -1


def _extract_entities(text: str) -> list[str]:
    """Extract entity mentions from text.

    Simple heuristic: capitalized multi-word phrases, quoted strings, numbers,
    and date-like patterns.  Good enough for entity linking within a session.
    """
    entities: list[str] = []

    # Quoted strings
    for m in re.finditer(r'"([^"]{2,60})"', text):
        entities.append(m.group(1).strip())

    # Capitalized phrases (2+ words starting with uppercase)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|de|von|van|le|la|del|al))+)\b', text):
        phrase = m.group(1).strip()
        if len(phrase) > 3 and phrase not in {"The", "This", "That", "These", "Those"}:
            entities.append(phrase)

    # Single capitalized words that look like proper nouns (not sentence starters)
    # Only grab these if preceded by non-sentence-boundary context
    for m in re.finditer(r'(?<=[a-z,;:]\s)([A-Z][a-z]{2,})\b', text):
        word = m.group(1)
        if word not in {"The", "This", "That", "However", "Also", "Additionally",
                        "Furthermore", "Moreover", "Therefore", "Because", "Although"}:
            entities.append(word)

    # Years (4-digit numbers that look like years)
    for m in re.finditer(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', text):
        entities.append(m.group(1))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def _normalize_answer(answer: str) -> str:
    """Normalize a candidate answer for matching."""
    return re.sub(r'\s+', ' ', answer.strip().lower())


class FactRegistry:
    """Structured evidence accumulator for research sessions.

    Maintains two collections:
    - facts: extracted information from documents, tagged with entities and provenance
    - candidates: proposed answers with confidence scores that update with evidence

    Designed to be injected into the agent's REPL namespace alongside
    search(), get_document(), and llm_query().
    """

    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._candidates: list[Candidate] = []
        self._entity_index: dict[str, list[int]] = {}  # entity_lower -> fact indices
        self._iteration: int = 0

    def set_iteration(self, n: int) -> None:
        """Called by the harness to track which iteration we're on."""
        self._iteration = n

    # ----- Fact management ----- #

    def add(self, text: str, source: str = "", kind: str = "fact") -> str:
        """Add extracted fact(s). Multi-line text is split into individual facts.

        Returns a short confirmation string (for REPL output).
        """
        lines = [ln.strip().lstrip("•-*") .strip()
                 for ln in text.strip().split("\n") if ln.strip()]

        added = 0
        for line in lines:
            if len(line) < 5:
                continue
            entities = _extract_entities(line)
            fact = Fact(
                text=line,
                source=source,
                kind=kind,
                entities=entities,
                iteration=self._iteration,
            )
            idx = len(self._facts)
            self._facts.append(fact)
            for ent in entities:
                key = ent.lower()
                self._entity_index.setdefault(key, []).append(idx)
            added += 1

        return f"Added {added} fact(s) from {source or 'unknown'}. Total: {len(self._facts)} facts."

    def for_entity(self, entity: str) -> list[str]:
        """All facts mentioning a specific entity."""
        key = entity.lower()
        indices = self._entity_index.get(key, [])
        return [f"[{self._facts[i].source}] {self._facts[i].text}" for i in indices]

    def entities(self) -> list[str]:
        """All known entities, sorted by frequency (most mentioned first)."""
        counts: dict[str, int] = {}
        # Use the original case from the first occurrence
        canonical: dict[str, str] = {}
        for fact in self._facts:
            for ent in fact.entities:
                key = ent.lower()
                counts[key] = counts.get(key, 0) + 1
                if key not in canonical:
                    canonical[key] = ent
        return [canonical[k] for k, _ in sorted(counts.items(), key=lambda x: -x[1])]

    # ----- Candidate management ----- #

    def propose(self, answer: str, confidence: float = 0.5,
                evidence: str = "") -> str:
        """Propose a candidate answer with confidence score.

        If the candidate already exists, updates confidence to the max.
        """
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c.answer) == norm:
                c.confidence = max(c.confidence, confidence)
                if evidence:
                    c.evidence.append(evidence)
                return f"Updated '{c.answer}' → confidence {c.confidence:.2f} ({len(c.evidence)} evidence items)"

        cand = Candidate(
            answer=answer.strip(),
            confidence=min(max(confidence, 0.0), 1.0),
            evidence=[evidence] if evidence else [],
            iteration=self._iteration,
        )
        self._candidates.append(cand)
        return f"Proposed '{cand.answer}' with confidence {cand.confidence:.2f}"

    def reinforce(self, answer: str, evidence: str = "") -> str:
        """Increase confidence for a candidate (+{REINFORCE_ALPHA})."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c.answer) == norm:
                c.confidence = min(c.confidence + REINFORCE_ALPHA, 1.0)
                if evidence:
                    c.evidence.append(evidence)
                return f"Reinforced '{c.answer}' → {c.confidence:.2f}"
        # Auto-propose if not found
        return self.propose(answer, confidence=0.5 + REINFORCE_ALPHA, evidence=evidence)

    def contradict(self, answer: str, reason: str = "") -> str:
        """Decrease confidence for a candidate (-{CONTRADICT_ALPHA})."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c.answer) == norm:
                c.confidence = max(c.confidence - CONTRADICT_ALPHA, 0.0)
                if reason:
                    c.evidence.append(f"[CONTRADICTED] {reason}")
                return f"Contradicted '{c.answer}' → {c.confidence:.2f}"
        return f"No candidate '{answer}' found to contradict."

    def weaken(self, answer: str, reason: str = "") -> str:
        """Slightly decrease confidence for a candidate (-{WEAKEN_ALPHA})."""
        norm = _normalize_answer(answer)
        for c in self._candidates:
            if _normalize_answer(c.answer) == norm:
                c.confidence = max(c.confidence - WEAKEN_ALPHA, 0.0)
                if reason:
                    c.evidence.append(f"[WEAKENED] {reason}")
                return f"Weakened '{c.answer}' → {c.confidence:.2f}"
        return f"No candidate '{answer}' found to weaken."

    def candidates(self) -> str:
        """Ranked candidate answers with confidence and evidence count."""
        if not self._candidates:
            return "No candidates proposed yet."
        sorted_cands = sorted(self._candidates, key=lambda c: -c.confidence)
        lines = ["Candidate answers (ranked by confidence):"]
        for i, c in enumerate(sorted_cands):
            marker = "→" if i == 0 else " "
            lines.append(
                f"  {marker} {c.answer} (confidence: {c.confidence:.2f}, "
                f"evidence: {len(c.evidence)} items)"
            )
        return "\n".join(lines)

    def best_candidate(self) -> tuple[str, float] | None:
        """Highest-confidence candidate answer, or None."""
        if not self._candidates:
            return None
        best = max(self._candidates, key=lambda c: c.confidence)
        return (best.answer, best.confidence)

    # ----- Summaries ----- #

    def summary(self) -> str:
        """Deduplicated, entity-grouped fact summary."""
        if not self._facts:
            return "No facts collected yet."

        lines = [f"=== Fact Summary ({len(self._facts)} facts, "
                 f"{len(self._entity_index)} entities) ===\n"]

        # Group by entity (show top entities by frequency)
        top_entities = self.entities()[:10]
        shown_fact_ids: set[int] = set()

        for ent in top_entities:
            key = ent.lower()
            indices = self._entity_index.get(key, [])
            if not indices:
                continue
            lines.append(f"[{ent}]")
            for idx in indices[:5]:  # max 5 facts per entity
                f = self._facts[idx]
                lines.append(f"  • {f.text} (from {f.source})")
                shown_fact_ids.add(idx)
            if len(indices) > 5:
                lines.append(f"  ... and {len(indices) - 5} more")
            lines.append("")

        # Show ungrouped facts
        ungrouped = [i for i in range(len(self._facts)) if i not in shown_fact_ids]
        if ungrouped:
            lines.append("[Other facts]")
            for idx in ungrouped[:10]:
                f = self._facts[idx]
                lines.append(f"  • {f.text} (from {f.source})")
            if len(ungrouped) > 10:
                lines.append(f"  ... and {len(ungrouped) - 10} more")

        # Append candidates
        if self._candidates:
            lines.append("")
            lines.append(self.candidates())

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_facts = len(self._facts)
        n_cands = len(self._candidates)
        n_ents = len(self._entity_index)
        best = self.best_candidate()
        best_str = f", best='{best[0]}' ({best[1]:.2f})" if best else ""
        return f"FactRegistry({n_facts} facts, {n_ents} entities, {n_cands} candidates{best_str})"
