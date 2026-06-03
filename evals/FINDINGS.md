# Eval findings — problems & improvement suggestions

Issues surfaced by the eval suite, each with a suggested fix. These are real
system problems (or eval-methodology notes), **not** test bugs. Raw metric
snapshots live in `evals/report/`. Remove an item here once it's fixed.

Severity: 🔴 data loss / correctness · 🟠 quality/perf · 🟡 cosmetic / methodology.

---

## 🔴 0. CombinerAgent crashed on every query (`max_tokens` rejected) — FIXED

The first E2E run hit `openai.BadRequestError 400: 'max_tokens' is not supported
with this model. Use 'max_completion_tokens' instead.` from `CombinerAgent`. The
combiner writes every answer, and the configured **gpt-5.4** (GPT-5 family) rejects
`max_tokens`, so `run_multi_agent_query` / `python -m src.main query` raised for
**every** question. The other agents don't set `max_tokens`, so only the final
answer step was broken — easy to miss without an E2E eval.

- **Fixed** in `src/rag/agents/combiner_agent.py`: pass the cap as
  `model_kwargs={"max_completion_tokens": settings.llm_max_tokens}` instead of
  `max_tokens=`. (Routed via `model_kwargs` so it reaches the API regardless of the
  langchain-openai version.)
- Follow-up worth checking: any other GPT-5-only parameter constraints (e.g.
  temperature) on the chat model, and whether to pin a langchain-openai version
  that maps this automatically.

## 🔴 1. Citation regex misses the nominative `νόμος`

`Ο νόμος 5167/2024 …` produces **no candidate**, so the chunk never reaches the
citation classifier — a permanent missing edge in the citation graph.

- Evidence: `datasets/regex_candidates.jsonl` → `nu_word_nom` (tagged `known_gap`).
- Cause: the `_LAW_REF_RE` alternation has `νόμου?/νόμο/νόμοι/νόμων` but not `νόμος`.
- **Fix:** add `νόμος` (and any other missing nominatives) to the alternation in
  `src/citations/extractor.py`. Then drop the `known_gap` flag so the eval gates it.

## 🔴 2. Citation regex misses genitive `προεδρικού διατάγματος`

Common phrasing "διατάξεις **του προεδρικού διατάγματος** Χ/Υ" is dropped.

- Evidence: `datasets/regex_candidates.jsonl` → `pd_word` (tagged `known_gap`).
- Cause: the regex only matches `προεδρικ[όή]`, not the genitive `προεδρικού`.
- **Fix:** broaden the Π.Δ. word-form branch to cover `προεδρικ\w*` (or list the
  genitive forms). Remove the `known_gap` flag afterward.

> Items 1–2 are the highest-value fixes: every missed candidate is permanent
> citation-graph data loss (a chunk with no regex hit is never reclassified).

## 🔴 3. "Hybrid" retrieval is silently pure-semantic for questions

The §2.4 ablation showed **byte-identical** metrics across all 7 retrieval configs
(semantic-only, fts-only, every weight ratio, every `rrf_k`):
`doc_recall@10=0.82, exact_recall@10=0.70, mrr=0.6322`. Adversarial verification
(`hybrid_search` weight overrides vs the pure `match_documents` RPC) showed the
hybrid result **exactly equals pure cosine** for every question.

- Cause A — `match_documents_hybrid` builds its lexical leg with
  `websearch_to_tsquery('simple', …)`, which **ANDs all query terms**. A long
  natural-language question becomes a many-term conjunction that **no chunk
  satisfies → 0 FTS rows**, leaving only the semantic leg in the full-outer-join.
- Cause B — the `'simple'` config does **no Greek stemming**, so even short
  paraphrased terms must match the stored inflection verbatim. (A literal lifted
  phrase *does* match and reorder results — proving the index/leg work; it's the
  query construction that fails on real questions.)
- Consequences:
  - Tuning `full_text_weight` / `semantic_weight` / `rrf_k` is **inert** for the
    ChunkAgent until the lexical leg fires. The 1/1/50 defaults are neither right
    nor wrong — they do nothing for question retrieval.
  - The FTS leg only contributes for short verbatim keyword queries (exact ΦΕΚ
    numbers, ListingAgent scoped searches) — not RAG questions.
  - The **semantic embedding does 100% of the work** for question retrieval, so the
    embedding model is the real recall lever (see `embed_ablation.py`).
- **Fix options:**
  - OR the lexical terms instead of AND (build `to_tsquery` with `|`, or extract
    keywords — `ListingAgent._query_keywords` already exists — and OR those) so
    partial lexical matches re-enter the fusion.
  - Add Greek stemming (a Greek text-search config / Snowball Greek dictionary) so
    inflected query terms match stored forms.
  - Guardrail: `websearch_to_tsquery('simple', 'άρθρο')` on a ubiquitous single
    token **times out** (`statement_timeout`) — any OR rewrite must keep the
    candidate scan bounded.

## 🟠 4. ~6% of questions don't retrieve the right document

On the **quality-gated** 50-question gold set, document-level `recall@10 = 0.94`,
`recall@50 = 0.96` (the earlier 0.82 was depressed by garbled questions — see the
methodology note). Crucially, the headroom from k=10→50 is only **0.02**, so the
`least(match_count*2, 50)` candidate cap and HNSW `ef` are **not** the bottleneck —
the residual misses are genuine embedding misses, which points back at the model.

- Driver confirmed: the embedding model. Since the lexical leg is dead (#3), the
  semantic embedding carries everything, and `embed_ablation.py` (mini-index of
  1550 chunks, semantic-only) shows a clear lift from a stronger model:

  | model | doc@1 | doc@5 | doc@10 | exact@10 | MRR |
  |---|---|---|---|---|---|
  | text-embedding-3-small (current) | 0.76 | 0.78 | 0.80 | 0.78 | 0.74 |
  | **text-embedding-3-large** | **0.86** | **0.88** | **0.88** | **0.84** | **0.82** |

  +8–10 pts document recall, +0.08 MRR. (Absolute numbers are inflated by the small
  mini-index; the *relative* gap is the signal.)
- **Recommended fix:** upgrade the corpus embedding to `text-embedding-3-large`.
  Trade-offs to weigh against the scaling constraint (8 GB RAM, target ~1M chunks):
  - 3072-dim vs 1536-dim → **2× vector storage + RAM** (1M × 3072 × 4 B ≈ 12 GB raw)
    and **~6.5× embedding cost** ($0.13 vs $0.02 / 1M tokens).
  - Requires re-embedding the whole corpus + migrating the `vector(1536)` column to
    `vector(3072)` and rebuilding the HNSW index.
  - **Mitigation to test next:** `text-embedding-3-large` supports the `dimensions`
    param — request 1536-dim large vectors to keep storage flat while capturing most
    of the quality gain. Add a third row to `embed_ablation.py` to measure it.
- Also fix the FTS leg (#3) so a second, independent signal returns, then
  re-measure end-to-end with `ablation.py`.

## 🟠 5. Retrieval latency ~5 s/query (p50 4.83 s, p95 5.48 s)

Measured by `scale/scale_probe.py` over 50 queries at 57k chunks (k=50). High
already, and HNSW + remote round-trip both grow with the corpus.

- **Fix / track:** watch the p50/p95 curve as the corpus grows; investigate HNSW
  `ef_search`, connection reuse, and whether the self-hosted Supabase round-trip
  dominates. Re-run the probe at each scale tier.

## 🔴 8. RAM wall: the ~1M-chunk target won't fit in 8 GB

`scale_probe.py` RAM projection (float32 vectors only):

| | vectors @1M | + HNSW graph (×1.5–3) |
|---|---|---|
| 1536-dim (current) | 6.14 GB | ~9–18 GB |
| 3072-dim (`-3-large`) | 12.29 GB | ~18–37 GB |

Either way the **8 GB box is exceeded well before 1M chunks**, before Postgres + OS
overhead. This directly couples to #4: upgrading to `text-embedding-3-large` for
recall **doubles** the memory problem.

- **Options to evaluate (none are a quick edit):** dimension-reduced large vectors
  (`dimensions=1536`), scalar/binary quantization (pgvector `halfvec`/`bit`), table
  partitioning / sharding by year, or moving the index off the 8 GB box. Add a
  `dimensions=1536` row to `embed_ablation.py` to see if reduced-dim large keeps the
  recall gain at flat storage.

## 🟠 9. Year-filtered retrieval intermittently times out

A `metadata @> {"year": …}` filtered query hit the DB statement timeout (`57014`)
on the first probe run; a 10-query re-run had 0 timeouts at ~4.5 s each. pgvector
HNSW **post-filters** (find neighbours, then drop non-matches), so the year filter
can't use the vector index efficiently — latency sits near the timeout ceiling and
will worsen at scale. The ChunkAgent uses this filter for `query --year`.

- **Fix (future):** a partial/composite index strategy for year-scoped search, or
  pre-filtering via a year-partitioned table. Track filtered timeout rate in the
  scale probe.

## 🟠 7. Combiner occasionally over-claims beyond its sources

§5 faithfulness mean is healthy (**0.92** over 12 answerable questions, Greek-rate
and citation-validity 1.0), but 2/12 answers asserted specifics the cited chunks
don't fully support (faithfulness 0.5 and 0.67 — e.g. naming a specific directorate
the source only implies). Not systemic, but it's the failure mode that matters most
for legal text.

- **Fix (needs its own loop, not a quick edit):** tighten the CombinerAgent system
  prompt toward "state only what the excerpts explicitly say; attribute carefully,"
  then re-run `test_faithfulness.py` to confirm no regression and watch for
  over-abstention. Track `faithfulness_min`, not just the mean.

## 🟡 11. Relational routing misses generic-verb phrasings (~1 in 15)

§2.3 over 15 real focal laws: `routing_fired_rate = 0.93`, and when routing fires
`edge_recall = 0.99` (the citation-graph differentiator works well end-to-end). The
one miss (`Ν.4878/2022`, query "Ποιους νόμους **επηρεάζει** ο ν. 4878/2022;") did
not trigger relational intent — generic/implicit relational verbs (επηρεάζει,
σχετίζεται) are weaker triggers than the closed-vocabulary verbs
(τροποποιεί/καταργεί/...).

- **Suggestion (needs re-eval, not a blind edit):** add a generic-verb example to
  `RelationAgent._INTENT_USER_TEMPLATE` mapping "επηρεάζει/σχετίζεται" → relational,
  outgoing, relations=null; then re-run `evals/graph` and the §4 routing eval to
  confirm no regression (don't over-trigger on non-relational content questions).

## 🟡 10. The extraction-quality gate can't detect garbled text (by design)

`assess_text_quality` judges char count / chars-per-page / page coverage, not
readability — so a PDF with a dense-but-garbled text layer passes it. This is the
blind spot that motivated always-OCR; it's now pinned by
`evals/component/test_text_quality.py` (incl. a test asserting garbled-but-dense
text still PASSES, so the rationale isn't silently lost).

- **Suggestion (only if the text layer is ever trusted again):** add a Greek-ratio
  gate using `greek_char_count`/`latin_char_count` (now in `src/ingestion/quality.py`)
  — the tests show clean Greek scores >0.8 while mojibake scores <0.2.

## 🟡 6. `_strip_invalid_citations` leaves a double space

Removing a mid-sentence `[n]` leaves "και&nbsp;&nbsp;το" — it only collapses
whitespace *before punctuation*. Cosmetic; currently locked as expected behaviour
in `test_citation_format.py`.

- **Fix (optional):** also collapse the resulting double space when a citation is
  stripped mid-sentence.

---

## Methodology notes

- **Gold sets undercount with single-exact-chunk labels** on this corpus
  (near-duplicate sibling chunks). The retrieval harness reports both exact and
  document-level recall and **gates on document-level**.
- **Sampling/generation skew is easy to introduce.** The first synthetic set
  (`--max-scan 4000`) drew 13 of its "2023" questions from just 2 table-heavy PDFs
  with near-unretrievable "code 1 = which gull?" lookups, depressing recall. Fixed
  by capping `--per-source` and forbidding self-referential/table-lookup questions
  in the generator prompt. Grow the set and add a few **real** user queries before
  ratcheting the recall floors up.
- **Corpus context for these numbers:** 57,268 chunks, years 2022 (89%) + 2023
  (11%) only.
