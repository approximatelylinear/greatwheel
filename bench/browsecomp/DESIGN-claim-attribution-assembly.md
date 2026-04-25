# Design: Claim-Level Attribution via Structured Belief Assembly

**Status:** Proposed
**Date:** 2026-04-25
**Companion to:** `DESIGN-entity-aboutness-rerank.md` (retrieval side)

---

## Motivation

`DESIGN-entity-aboutness-rerank.md` solves the *retrieval* side of citation
quality: surface the documents that are focally about the query entity, not
just the ones that mention it. This doc tackles the **assembly** side: given
a candidate set of docs (ideally including legitimate context — Pepin and
Charles Martel docs *should* be in the set when answering "who was
Charlemagne"), how do we tie each *claim* in the generated answer back to
the *specific source span(s)* that justify it?

The naive RAG pattern — retrieve → stuff context → generate prose → cite
post-hoc — produces hallucinated citations and aboutness-leak on the
attribution layer. A claim like "Charlemagne was crowned in 800" gets
attributed to a Pepin-bio sentence that *mentions* Charlemagne becoming
emperor, because that sentence is semantically similar to the claim, even
though it does not entail the date. Same failure mode as retrieval-side
aboutness, on a finer scale.

The fix is to make citations a **structural property** of the answer, not a
post-hoc lookup. We borrow the architecture from BLF (Murphy 2026,
arXiv:2604.18576): a structured belief object, updated iteratively, where
every claim is born with its source slots attached. BLF's ablation shows
that removing the structured belief state degrades performance as much as
removing web search entirely — structure-of-evidence is as load-bearing as
evidence itself.

## The five sub-problems

1. **Similarity is not entailment.** Embedding cosine and BM25 measure
   topical overlap, which is symmetric. Citation requires entailment, which
   is directional ("does the span imply the claim?"). The standard
   attribution mistake is using similarity as a proxy.

2. **Claim atomicity.** "Charlemagne, son of Pepin the Short, was crowned in
   800 by Pope Leo III" is four atomic claims in one sentence. Different
   sources may attest different sub-claims. Without decomposition, you cite
   one source for a sentence that's only partially supported by it.

3. **Synthesis claims have no single source.** "Charlemagne unified Western
   Europe" is an aggregation. There is no passage to cite directly. These
   are the claims that hallucinate citations because the model has to
   attach *something*.

4. **Implicit entailment.** Sources rarely state things in the model's
   preferred phrasing. "Pepin's heir" implicitly entails "son of Pepin." NLI
   models handle this brittlely; LLM judges handle it better at higher cost.

5. **Aboutness on the fine scale.** Even with the right docs in hand, a
   per-claim attributor that uses semantic similarity will frequently
   attribute Charlemagne claims to passages from the Pepin doc that mention
   Charlemagne, because those passages cluster tightly around the entity.
   The aboutness problem reappears at the span level.

## Hypothesis

A pipeline of (a) atomic claim decomposition, (b) fine-grained per-claim
re-retrieval at the proposition level, (c) NLI-based entailment scoring
(not similarity) for span-claim matching, and (d) a BLF-shaped structured
belief object that holds the answer-in-progress, will produce
materially-better citation precision than post-hoc span-matching, with
hallucinated citations approaching zero.

The structural claim is stronger than the empirical one: with this
architecture, *the only way for the LLM to add a claim is to attach its
supporting evidence*. There is no surface for a hallucinated citation to
hide on, because the citation is part of the data structure rather than a
generated string.

## Design

### The structured belief object

Per question, the LLM maintains a single JSON object updated iteratively as
evidence arrives:

```json
{
  "question": "Who was Charlemagne?",
  "claims": [
    {
      "id": "c1",
      "text": "Charlemagne was King of the Franks from 768 to 814",
      "confidence": "high",
      "supporting_evidence": [
        {"source_id": "wiki_charlemagne", "span_id": "s_47",
         "span_text": "Charlemagne reigned as King of the Franks...",
         "entailment_score": 0.94, "focal_score": 0.91},
        {"source_id": "encyclopedia_carolingian", "span_id": "s_12",
         "span_text": "...", "entailment_score": 0.87, "focal_score": 0.62}
      ],
      "counter_evidence": []
    },
    {
      "id": "c2",
      "text": "Charlemagne was the son of Pepin the Short",
      "confidence": "high",
      "supporting_evidence": [...]
    },
    {
      "id": "c3",
      "text": "Charlemagne unified Western Europe",
      "confidence": "synthesis",
      "supporting_evidence": [
        {"source_id": "wiki_charlemagne", "span_id": "s_91", ...},
        {"source_id": "wiki_saxon_wars", "span_id": "s_3", ...},
        {"source_id": "wiki_lombardy_conquest", "span_id": "s_8", ...}
      ],
      "counter_evidence": []
    }
  ],
  "open_questions": [
    "Was Charlemagne literate? Sources disagree."
  ],
  "unsupported_drafted": [
    "Charlemagne founded the modern European Union (drafted but no entailing evidence)"
  ]
}
```

Three things to notice:

- **Citations are structural, not generated.** Each claim has a list of
  evidence slots. The render step turns this object into prose; citations
  in the prose come from the slots, not from the LLM's free-form output.
- **`confidence: "synthesis"`** is a first-class status. Synthesis claims
  retain a list of *constituent* evidence (each entailing some sub-aspect)
  rather than pretending to be directly attestable. The render step can
  flag these as editorial syntheses.
- **`unsupported_drafted`** is the explicit hallucination defense. If the
  draft step produced a claim and no fine-grained retrieval pass found
  entailing evidence, the claim moves here rather than into the output.

### The pipeline

```
1. Candidate-doc retrieval
     [existing pipeline, ideally with the aboutness reranker from
      DESIGN-entity-aboutness-rerank.md applied]
   → top-K docs (K ≈ 20)

2. Draft answer generation
     LLM(question, candidate_docs) → draft_text
   [unchanged from current pipeline; we treat this as a "what does the
    LLM think the answer looks like" pass, not as the final output]

3. Atomic claim decomposition
     LLM(draft_text) → [claim_1, claim_2, ...]
   [structured-output prompt; ~1 cheap LLM call. Each claim is one
    atomic proposition. FActScore-style decomposition.]

4. Per-claim fine-grained re-retrieval
     For each claim:
       spans = proposition_index.search(claim.text, k=20,
                                        restrict_to=candidate_doc_ids)
   [proposition-level index over the candidate doc set. We re-search
    here rather than reusing chunk-level retrieval because chunks are
    too coarse for span-level attribution.]

5. Entailment scoring
     For each (claim, span) pair:
       e = MiniCheck(claim.text, span.text)
       a = focal_score(span)        # from doc-level primary_topics
   [MiniCheck-Flan-T5-Large or similar small NLI model. We score
    entailment, not similarity. The focal_score is the same one used
    in the retrieval-side reranker — reused as a feature here.]

6. Belief object construction
     For each claim:
       supporting = [(s, e, a) for s in spans if e > 0.5]
       if supporting is empty:
         move claim to unsupported_drafted
       elif len(supporting) >= 3:
         claim.confidence = "high"
       elif len(supporting) >= 1:
         claim.confidence = "medium"
       if claim.is_synthesis():           # detected via decomposition step
         claim.confidence = "synthesis"

7. Render
     LLM(belief_object) → final_answer
   [prompt: "Render this belief object into a fluent answer.
    Each sentence must carry citations to the source_ids in its
    supporting_evidence. Do not include claims from
    unsupported_drafted. Mark synthesis claims as such."]
```

Steps 3-7 are roughly equivalent to BLF's iterative belief-update loop
collapsed into a single non-iterative pass. We're not (yet) doing
sequential evidence gathering across multiple search rounds — that's a
Phase 4 extension once the single-pass version is working.

### Components

| File                                        | Role                                                                     |
|---------------------------------------------|--------------------------------------------------------------------------|
| `assembly/decomposer.py`                    | LLM call: draft → atomic claims (with synthesis-flag detection).         |
| `assembly/proposition_index.py`             | Per-doc proposition-level index (built lazily over candidate doc set).   |
| `assembly/entailment_scorer.py`             | MiniCheck wrapper. Batched (claim, span) → entailment score.             |
| `assembly/belief_builder.py`                | Assembles the structured belief object from claims + scored spans.       |
| `assembly/renderer.py`                      | LLM call: belief object → fluent cited answer.                           |
| `retrieval_benchmark_v2.py` (`+assembly`)   | End-to-end mode: retrieval → assembly → graded output.                   |

The proposition index is built **at query time over the candidate doc set**,
not over the full corpus. For 20 docs of ~2000 tokens each, that's ~40K
tokens to split into propositions — one cheap LLM pass per query. The
alternative (pre-building a proposition index over the whole corpus) is
~50x more storage and only worth it if assembly latency becomes the
bottleneck.

### Why MiniCheck

[MiniCheck](https://github.com/Liyan06/MiniCheck) (Tang et al. 2024) is the
SOTA small entailment scorer: their 7B variant rivals GPT-4 on
fact-checking benchmarks (LLM-AggreFact); the Flan-T5-Large variant (770M)
is still strong and runs at ~10ms/pair on a single GPU. AlignScore is a
reasonable alternative.

What we're *not* using: the model's own attention/output (Anthropic
Citations API style). Two reasons:

1. We're constrained to local-Ollama models, and Ollama doesn't expose the
   structured-citation output that the Citations API does.
2. Even where it's available, generation-time citation depends on the
   generator model's training. A separate NLI model is a stronger,
   model-independent guarantee.

If/when we have access to a citations-aware generation model, the NLI step
becomes a *verifier* on top rather than the only line of defense.

## Integration with the retrieval-side reranker

The two designs compose:

- The retrieval-side aboutness reranker improves the *candidate set* the
  assembly stage works with. With it, the candidate set is more likely to
  include the actual focal doc, so per-claim re-retrieval has good spans
  to find.
- The assembly side tolerates a noisier candidate set than naive RAG does,
  because the entailment step rejects spans that are merely topical. So
  even without the retrieval-side fix, assembly improves citation
  precision — but with both, both metrics move.

The `focal_score` field stored per doc by the retrieval-side design is
reused as a feature in the assembly stage's evidence ranking — so the
ingest-time entity-extraction work pays off in both places.

## Evaluation

### Citation precision and recall (the right metric)

For each generated answer, label per-claim:

- **Citation precision**: fraction of cited (claim, span) pairs where the
  span actually entails the claim, judged by a strong LLM (Claude or
  human spot-check).
- **Citation recall**: fraction of supportable claims that received a
  citation. Measures whether the system is conservatively dropping things
  it could have supported.
- **Hallucinated-citation rate**: fraction of cited spans that don't
  entail any claim in the answer (i.e., the model attached a span to make
  the citation slot non-empty).
- **Synthesis-flag accuracy**: fraction of `synthesis` claims that humans
  agree are genuinely synthesis (vs directly attestable but mislabeled).

These map onto [ALCE](https://github.com/princeton-nlp/ALCE)'s citation
precision/recall framework directly. We use ALCE's eval scripts where
applicable.

### Build claim-attribution-eval-30

Companion to focal-eval-30 from the retrieval-side design. Curate 30-50
questions where:

- The answer is non-trivial (multi-claim, possibly synthesis-heavy).
- Gold per-claim source spans exist (hand-labeled).

Run the assembly pipeline and grade against gold. This is the only metric
that will reveal whether the pipeline is doing what it claims.

### End-to-end agent eval on BrowseComp

For BrowseComp specifically: aboutness-correct citations should also help
the agent's *own* reasoning, because better-attributed evidence is easier
for the LLM to chain over. Run sample30 with assembly-on vs assembly-off
and measure agent accuracy. Hypothesis: modest (+2-5pp) accuracy lift on
top of any retrieval-side improvements, larger lift on questions where
the failure mode was "agent cited the wrong source and then reasoned from
it."

## Run plan

### Phase 0 — Build claim-attribution-eval-30 (1-2 days)

Curate 30-50 multi-claim questions over BrowseComp-Plus corpus with
hand-labeled per-claim source spans. Without this, no later phase has a
metric.

### Phase 1 — Decomposer + proposition index (2 days)

1. Implement `decomposer.py`. Validate on 30 hand-checked decompositions:
   does it produce atomic, non-overlapping claims? Does it correctly flag
   synthesis claims?
2. Implement `proposition_index.py`. For 20-doc candidate sets, build the
   index in <2s per query.

**Decision gate:** does manual inspection show ≥80% atomic-claim quality
on the validation set? If <80%, iterate on the decomposition prompt
before proceeding.

### Phase 2 — Entailment scorer integration (1 day)

1. Stand up MiniCheck-Flan-T5-Large as a service (or in-process if VRAM
   allows alongside other models).
2. Validate on a small (claim, span) pair set with hand-labeled
   entailment. Calibrate the entailment threshold (default 0.5; may need
   adjustment per the model's score distribution on our corpus).

### Phase 3 — Belief builder + renderer + end-to-end pipeline (2 days)

1. Wire steps 1-7 of the pipeline.
2. Run on claim-attribution-eval-30. Measure citation precision, recall,
   hallucinated-citation rate, synthesis-flag accuracy.

**Decision gate (the critical one):** citation precision ≥85% AND
hallucinated-citation rate ≤5%? If yes, the architecture works.
If precision is high but recall is low, we're being too conservative —
lower entailment threshold or expand fine-grained retrieval k. If
hallucinated-citation rate is high, the renderer LLM is ignoring the
structural constraint — strengthen the prompt or constrain output via a
schema.

### Phase 4 — End-to-end BrowseComp sample30 eval (1 day)

Run sample30 with assembly-on (and the retrieval-side reranker also on,
for the strongest candidate-set quality). Compare against:

- baseline (no aboutness reranker, no assembly)
- aboutness reranker only
- assembly only
- both

This 2x2 isolates the contribution of each side.

### Phase 5 — Iterative belief updates (optional, if Phase 3-4 land well)

The current design is a single-pass version of BLF: one decomposition,
one re-retrieval, one render. The full BLF pattern adds iterative search:
when a claim has insufficient evidence, the system issues a new search
to find better support, updates the belief, repeats. This is a meaningful
extension but only worth it if the single-pass version is shipping.

BLF's published evidence: iterative beats batch by ~3.8 BI points
(p<0.001). Same delta as removing search entirely. Strong prior that the
extension will pay off.

## Risks

1. **Decomposition quality dominates everything.** If the decomposer
   produces non-atomic or overlapping claims, every downstream stage
   degrades. Mitigation: validate decomposition against hand-checked
   examples in Phase 1 before building anything else on top.

2. **MiniCheck may not generalize to our corpus.** MiniCheck was trained
   on news/Wikipedia-style text; our corpus is mixed. Implicit entailment
   ("Pepin's heir" → "son of Pepin") is the standard NLI weakness.
   Mitigation: spot-check entailment scores during Phase 2; have an LLM
   judge as a fallback for borderline (0.4-0.6) cases.

3. **The renderer LLM may ignore the structural constraint.** Asked to
   render a belief object with citations, the LLM might drop citations,
   fabricate new ones, or misorder them. Mitigation: constrained
   structured-output rendering (emit citations as JSON, then
   post-process into prose), or a verifier pass that checks every
   citation in the output appears in the belief object.

4. **Synthesis claim detection is itself an attribution problem.** "Is
   this claim a synthesis or directly attestable?" is a judgment call.
   Mitigation: be permissive in flagging — better to mark a directly-
   attestable claim as synthesis (UX shows constituent sources) than to
   over-confidently single-source a synthesis.

5. **Latency budget.** Per query: 1 draft + 1 decomposition + 1
   proposition-index build + N×M entailment scores + 1 render =
   ~4-8 LLM/NLI calls plus span scoring. Estimated 5-15s per query
   end-to-end. Acceptable for batch eval, marginal for interactive use.
   Mitigation: cache proposition indexes per doc; batch entailment scoring;
   make decomposition and proposition-indexing parallel.

6. **The BLF inspiration may not transfer.** BLF works in a forecasting
   domain where the belief object is a probability and aggregation has
   well-defined semantics. For QA, the belief object is a set of claims
   with evidence — the analog is intuitive but unproven. Mitigation: the
   single-pass version (no iterative updates) is functionally equivalent
   to attribute-first-then-generate, which has independent published
   support — so we have a fallback architecture even if the iterative
   extension doesn't pan out.

## Dependencies

```toml
# bench/browsecomp/pyproject.toml additions
minicheck = ">=1.0"     # or load Flan-T5-Large directly via transformers
transformers = ">=4.40"
```

If MiniCheck doesn't expose a clean Python API, load
`lytang/MiniCheck-Flan-T5-Large` from HuggingFace directly.

## Success Criteria

| Outcome                                                              | Interpretation                                  | Next step                                                   |
|----------------------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------|
| Citation precision ≥85%, hallucination rate ≤5% on attrib-eval-30    | Architecture works as designed                  | Phase 4-5; productionize                                    |
| High precision, low recall                                           | Too conservative                                | Lower entailment threshold; expand fine-grained k           |
| High recall, low precision                                           | Renderer ignoring structural constraint         | Constrained output; verifier pass                           |
| Both flat                                                            | Decomposition or NLI is the bottleneck          | Ablate: hand-decompose; substitute LLM judge for NLI        |
| Citation metrics improve, sample30 accuracy flat                     | Citation quality decoupled from agent accuracy  | Ship for citation use case; don't claim agent-accuracy win  |
| Citation metrics improve AND sample30 accuracy lifts ≥3pp            | Better attribution improves agent reasoning     | Strong case for both designs together                       |

## References

- BLF: Murphy, K. (2026). *Agentic Forecasting using Sequential Bayesian
  Updating of Linguistic Beliefs.* arXiv:2604.18576.
- MiniCheck: Tang, L. et al. (2024). *MiniCheck: Efficient Fact-Checking
  of LLMs on Grounding Documents.* arXiv:2404.10774.
- Attribute First, Then Generate: Slobodkin, A. et al. (2024).
  arXiv:2403.17104.
- FActScore: Min, S. et al. (2023). *FActScore: Fine-grained Atomic
  Evaluation of Factual Precision in Long Form Text Generation.*
  arXiv:2305.14251.
- Dense X Retrieval / Propositionizer: Chen, T. et al. (2023).
  *Dense X Retrieval: What Retrieval Granularity Should We Use?*
  arXiv:2312.06648.
- ALCE: Gao, T. et al. (2023). *Enabling Large Language Models to Generate
  Text with Citations.* arXiv:2305.14627.
- Self-RAG: Asai, A. et al. (2023). arXiv:2310.11511.
