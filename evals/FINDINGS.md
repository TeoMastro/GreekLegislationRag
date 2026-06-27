# Eval findings — problems & improvement suggestions

Issues surfaced by the eval suite, each with a suggested fix. These are real
system problems (or eval-methodology notes), **not** test bugs. Raw metric
snapshots live in `evals/report/`. Remove an item here once it's fixed.

Severity: 🔴 data loss / correctness · 🟠 quality/perf · 🟡 cosmetic / methodology.

---

## 🔴 12. Every full-corpus query timed out at 105k chunks (HNSW skipped on empty filter) — FIXED & VERIFIED

Re-running the retrieval tier after the corpus doubled (57k → **105,441** chunks)
returned `doc_recall@10 = 0.0` with a **100% `statement_timeout` (57014) rate** —
every one of the 200 queries died at the ~8 s ceiling. Pure-semantic *and* hybrid
both failed; only the scoped per-PDF path (`known_item`) survived.

- **Cause:** `match_documents` and the hybrid semantic CTE carry `WHERE metadata
  @> filter`, and unscoped callers pass `filter = '{}'`. `metadata @> '{}'` matches
  every row and is GIN-indexable, so the planner built candidates from
  `documents_metadata_idx` (all 105k rows) then did an **exact distance sort — the
  HNSW index was never consulted**. This was always brute-force KNN; it merely fit
  under the 8 s timeout at ~57k (old scale p50 4.8 s) and crossed it at 2×.
  Proven by EXPLAIN: bare `ORDER BY embedding <=> q LIMIT 10` → Index Scan, 70 ms;
  the same query + `WHERE metadata @> '{}'` → seq/bitmap scan, times out.
  `ANALYZE documents` alone did **not** fix it (necessary for the bare-query plan,
  but the WHERE is the structural blocker).
- **Fixed** in `sql/005_hnsw_empty_filter.sql`: both RPCs rewritten in plpgsql with
  explicit `IF filter = '{}'` branches — empty filter runs a WHERE-less query the
  HNSW index serves; a real filter keeps the containment predicate (small candidate
  set ⇒ fast exact sort, the path that always worked). NB a `WHERE filter='{}' OR
  metadata @> filter` shortcut does **not** work: inside a cached plpgsql plan
  `filter` is a runtime parameter the planner can't fold to a constant, so it stays
  a runtime filter and brute-forces. Migration also re-asserts `service_role`
  `statement_timeout = 60s` (001 set it but it was not in effect on the self-hosted
  box — the role ran on the ~8 s default).
- **Verified (online, 2026-06-26):** post-005 probe — unfiltered semantic 0.57 s,
  full hybrid 3.86 s (cold), 0 timeouts. Retrieval tier: **timeout_rate 1.0 → 0.0**,
  `doc_recall@10` **0.0 → 0.75** (n=200), `known_item` 1.0/1.0. Scale probe
  unfiltered **p50 4.8 s → 0.98 s**.
- **Harness hardened:** `test_retrieval_recall.py` now catches 57014 per-row, counts
  it an evaluated doc-miss, and reports `timeouts` / `timeout_rate` — so a slow query
  degrades the metric instead of aborting the whole 200-row run with zero data.

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

## 🔴 1–2. Citation regex missed nominative `νόμος` and genitive `προεδρικού διατάγματος` — FIXED

`Ο νόμος 5167/2024 …` and "του **προεδρικού διατάγματος** Χ/Υ" produced **no
candidate**, so those chunks never reached the citation classifier — permanent
missing edges in the citation graph. Every missed candidate is permanent
data loss (a chunk with no regex hit is never reclassified), so these were the
highest-value fixes.

- **Fixed** in `src/citations/extractor.py`: the `_LAW_REF_RE` alternation now
  uses `νόμ\w*` (covers νόμος/νόμου/νόμο/νόμοι/νόμων) and
  `προεδρικ\w* … δι[αά]τ[αά]γμα\w*` (covers all προεδρικό/-ή/-ού + διάταγμα/
  διατάγματος inflections, accent-tolerant). The normalizer already had the
  matching aliases, so no change there.
- The `known_gap` flags were dropped from `nu_word_nom` and `pd_word` in
  `evals/datasets/regex_candidates.jsonl`, so the eval now gates these shapes.
  `test_regex_candidates.py`: `recall_supported=1.0`, `recall_overall=1.0`,
  `fp_rate=0.0`, all 14 formats at 1.0.

## 🔴 3. "Hybrid" retrieval is silently pure-semantic for questions — FIXED & VERIFIED

**Verified** by re-running §2.4 ablation (50 queries, k=10) after applying
`sql/003`. The configs are **no longer byte-identical** — the lexical leg fires:

| config (ftw/stw/rrf_k) | doc@1 | doc@5 | doc@10 | exact@10 | mrr |
|---|---|---|---|---|---|
| `fts_only` (1/0/50) | 0.18 | 0.38 | 0.46 | 0.14 | 0.0719 |
| `hybrid_balanced` (1/1/50, **old default**) | 0.52 | 0.90 | 0.92 | 0.84 | 0.4712 |
| **`hybrid_semantic_2x` (1/2/50, NEW default)** | **0.72** | 0.90 | **0.94** | **0.84** | 0.5832 |
| `semantic_only` (0/1/50) | 0.76 | 0.88 | 0.92 | 0.82 | 0.6427 |

`fts_only` went from ~0 / identical to a real independent signal (doc@10=0.46),
confirming the leg now contributes. No `57014` timeouts — the stopword guardrail held.

**Tuning result + action taken:** equal weighting (the old 1/1/50 default) let the
weaker lexical leg drag down top-rank precision (doc@1 0.76→0.52, ~12/50 queries).
`hybrid_semantic_2x` (1/2/50) **strictly dominates** 1/1/50 (≥ on every metric:
doc@1 +0.20, doc@10 +0.02, mrr +0.11) and beats pure-semantic on the metrics the
combiner depends on (doc@10 0.94, exact@10 0.84). **Production default changed to
`semantic_weight=2.0`** in `src/config.py`; `rrf_k` is inert (k20/k100 identical).

**Follow-up (small sample — n=50 synthetic):** a doc@1/MRR gap to pure-semantic
remains (0.72 vs 0.76; mrr 0.58 vs 0.64). `ablation.py` now includes
`semantic_3x`/`4x` rows — re-run to see if a higher ratio recovers doc@1/MRR while
keeping the doc@10/exact@10 edge. Grow the gold set + add real user queries before
ratcheting the recall floor up.

---

### Fix detail (Cause A + Cause B)

The lexical leg no longer builds its query
with `websearch_to_tsquery` (which ANDed every term). The caller now builds an
**OR-of-lexemes** `to_tsquery` string in `src/retrieval/store._build_fts_query`
and passes it as a new `query_fts` RPC arg:
- Distinctive terms are accent-folded, stopword-filtered, and OR'd (`a | b | …`)
  so any single lexical hit re-enters RRF fusion. (Originally each alpha term got a
  `:*` prefix for partial-inflection matching; **removed** once the `greek` config
  added real stemming — see the Cause B resolution above.)
- **Guardrail:** the ubiquitous terms that tripped `statement_timeout` (άρθρο,
  νόμος, παράγραφος, …) are in `_FTS_STOPWORDS`, and the term count is capped
  (`max_terms=6`), so the OR rewrite keeps the candidate scan bounded. Empty
  `query_fts` ⇒ pure-semantic (prior de-facto behaviour, preserved).
- SQL: `match_documents_hybrid` updated in `sql/001_init.sql` (fresh bootstrap)
  and `sql/003_hybrid_fts_or.sql` (migration for existing DBs — drops the old
  7-arg overload first, then recreates with `query_fts`).

**Resolved — Cause B (full stemming), via the `greek` text-search config.**
The `fts` column + the RPC's `to_tsquery` moved from `'simple'` (no stemming) to
`'greek'` (Snowball), so inflected query terms now match stored inflections in
*both* directions — not just the prefix-extends-query case the `:*` hack covered.
Verified on live data: of 100 chunks containing the inflected «νόμου», a «νόμος»
query matches **97 under `greek` vs 0 under `simple`**. `f_unaccent` is kept on
both sides (doesn't break stemming — 98/100 — and preserves accent-insensitivity).
The `:*` prefix in `_build_fts_query` was **removed** (stemming supersedes it).
- SQL: `sql/004_greek_fts_config.sql` (migration — drops & re-adds the generated
  `fts` column as `greek`, rebuilds the GIN index, repoints the RPC) + `sql/001`
  for fresh bootstrap. **Applied to the live DB** (57,268 rows regenerated).
- **NB:** this Postgres' `greek` config Snowball-stems but carries **no stopword
  dictionary** (verified: `to_tsquery('greek','και')` → `'κα'`, not dropped), so
  `_FTS_STOPWORDS` stays fully intact — it's still the only stopword mechanism and
  the timeout guardrail (ubiquitous domain terms like άρθρο still match ~everything).

**Post-greek ablation (50 queries, k=10) — greek lifts the lexical leg but it
still drags the fused result:** `fts_only` doc@10 **0.46→0.50** (stemming helps),
but `semantic_only` now **dominates every hybrid config** within this run
(doc@1 0.76 vs 0.66, doc@10 0.96 vs 0.94, exact@10 0.84 vs 0.82, mrr 0.645 vs 0.539;
hybrid wins only doc@5 by 0.02). The broader greek recall is ranked by IDF-less
`ts_rank_cd`, so fusing it costs top-rank precision. **Production weights left
unchanged** (`semantic_weight=2.0`): doc@10 is unchanged at 0.94, and this
question-only synthetic set excludes the keyword/ΦΕΚ-number/ListingAgent queries
where the lexical leg earns its keep — retuning on it would regress those.
*Watch-item:* if a grown gold set with real keyword/number queries still shows
semantic_only dominating, demote the lexical leg (or gate it to keyword-shaped
queries) rather than weighting it. (Caveat: `semantic_only` itself drifted
0.92→0.96 vs the pre-greek baseline, so the gold set isn't identical — trust the
within-run comparison, not the cross-table one.)

**To verify (not yet run — needs live DB + OpenAI):**
1. Apply `sql/003_hybrid_fts_or.sql` to the Supabase DB.
2. `python -m evals.retrieval.ablation` — confirm the configs are **no longer
   byte-identical** (fts_only returns rows; hybrid diverges from semantic_only),
   and watch for any year-filtered timeout regression (#9).

---

### Original diagnosis (kept for context)

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

## 🟠 5. Retrieval latency ~5 s/query (p50 4.83 s) — RESOLVED for unfiltered queries

Originally measured at 57k chunks (k=50): p50 4.83 s. That number was the
brute-force-KNN symptom of #12 — the unfiltered path never used HNSW. After
`sql/005` restored HNSW on the empty-filter path, the scale probe at **105k**
(2× the corpus) reports **unfiltered p50 0.98 s / p95 2.38 s** — a ~5× speedup
despite the larger haystack. Filtered (scoped) latency is now the outlier; see #9.

- **Track:** watch the unfiltered p50/p95 curve at each scale tier; the remaining
  cost is HNSW `ef_search` + the self-hosted Supabase round-trip.

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

## 🟠 9. Year-filtered retrieval is now the latency bottleneck (~4.6 s p50)

A `metadata @> {"year": …}` filter does not get the #12 fix: the filter is
non-empty, so `match_documents` takes the `else` branch and keeps the containment
predicate. A **year** filter is not selective (a year is ~20k of 105k chunks), so
the planner still does an exact distance sort over that large subset rather than
using HNSW — pgvector HNSW post-filters and can't index a non-selective predicate
efficiently. Scale probe at 105k: **filtered p50 4.6 s / p95 6.8 s** vs unfiltered
0.98 s, **0 timeouts** (under the 60 s ceiling 005 restored, but uncomfortably high).
Per-PDF scoped filters (`known_item`, ListingAgent) stay fast — that candidate set
is tiny. So the problem is specifically *broad* filters like year.

- **Fix (future):** a partial/composite index per year, year-partitioned tables, or
  raising HNSW `ef_search` with iterative-scan filtering (pgvector ≥0.8). Track
  `latency_filtered_s` + `timeouts_filtered` in the scale probe at each tier.

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

## 🟡 11. Relational routing misses generic-verb phrasings — FIXED & VERIFIED

§2.3 over 15 real focal laws: `routing_fired_rate = 0.93`, and when routing fires
`edge_recall = 0.99` (the citation-graph differentiator works well end-to-end). The
one miss (`Ν.4878/2022`, query "Ποιους νόμους **επηρεάζει** ο ν. 4878/2022;") did
not trigger relational intent — generic/implicit relational verbs (επηρεάζει,
σχετίζεται) are weaker triggers than the closed-vocabulary verbs
(τροποποιεί/καταργεί/...).

- **Fix** in `src/rag/agents/relation_agent.py`: `_INTENT_SYSTEM` now explicitly
  names generic/indirect relational verbs (επηρεάζει/σχετίζεται/συνδέεται/αφορά)
  alongside the closed-vocabulary ones, and `_INTENT_USER_TEMPLATE` gains a
  generic-verb worked example (`επηρεάζει` → relational, outgoing, relations=null).
  The negative content-question examples are kept to guard against over-triggering.
- Added two generic-verb rows to `evals/datasets/relation_intent.jsonl` (the exact
  `επηρεάζει` miss + a `σχετίζεται` "both" case) so §4.1 covers the gap (16 rows now).
- **Verified (online, 2026-06-07):** `RUN_ONLINE_EVALS=1 pytest evals/routing/test_relation_intent.py`
  → `is_relational_acc=1.0`, `is_relational_recall=1.0`, `focal_key_acc=1.0` (the two
  new generic-verb rows route as relational, zero routing misses). `evals/graph` (§2.3)
  → `routing_fired_rate` **0.93 → 1.0** (the `επηρεάζει` miss is gone),
  `edge_recall_mean_over_fired=1.0`, no false positives on content questions.
- **Follow-up surfaced & fixed during verification — direction on symmetric verbs.**
  The first online run cleared routing but failed `direction_acc` (0.7273 < 0.75 floor):
  3 misses, all sharing one root cause — symmetric/mutual relation verbs
  (`σχετίζεται`/`συνδέεται`/"τι σχέση έχει") should map to `direction="both"`, but the
  prompt gave the model no anchor (it had an `επηρεάζει→outgoing` example, none for
  symmetric→both), so it defaulted to a single direction. A second contributor: the old
  `incoming` description used "…αναφέρθηκαν σε έναν νόμο", biasing any `αναφέρεται` toward
  incoming even when the focal law is the grammatical *subject* (→ outgoing).
  **Fix** in `relation_agent.py`: the `direction` guidance now keys off the focal law's
  grammatical role (subject→outgoing, object→incoming) and adds an explicit
  symmetric/non-directional→`both` rule + a `συνδέεται→both` worked example (deliberately
  a different verb/law numbers than the gold rows, to avoid teaching-to-the-test).
  Re-run: `direction_acc` **0.7273 → 0.9091** (10/11; the lone residual is a temp-0
  flip on `αντικαθιστά`, well inside the floor). All four floors now pass.

## 🟡 10. The extraction-quality gate can't detect garbled text (by design)

`assess_text_quality` judges char count / chars-per-page / page coverage, not
readability — so a PDF with a dense-but-garbled text layer passes it. This is the
blind spot that motivated always-OCR; it's now pinned by
`evals/component/test_text_quality.py` (incl. a test asserting garbled-but-dense
text still PASSES, so the rationale isn't silently lost).

- **Suggestion (only if the text layer is ever trusted again):** add a Greek-ratio
  gate using `greek_char_count`/`latin_char_count` (now in `src/ingestion/quality.py`)
  — the tests show clean Greek scores >0.8 while mojibake scores <0.2.

## 🟡 6. `_strip_invalid_citations` leaves a double space — FIXED

Removing a mid-sentence `[n]` left "και&nbsp;&nbsp;το" — the function only
collapsed whitespace *before punctuation*, not internal.

- **Fixed** in `src/rag/agents/combiner_agent.py`: `_CITATION_RE` now captures
  any leading horizontal space (`[ \t]*\[(\d+)\]`), so dropping an invalid
  citation takes its preceding space with it (kept citations return `m.group(0)`
  unchanged). No blanket space-collapse, so intentional formatting is untouched.
- `test_citation_format.py` updated: the mid-sentence case now expects
  `"Πηγές [1] και το λένε."` (single space). 7 tests pass.

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
- **Corpus context for these numbers:** as of 2026-06-26, **105,441 chunks across
  5 years** (2022–2026); the citation graph holds 1,861 law_nodes / 24,081 edges.
  Gold sets were regenerated to span all 5 years (synthetic 200 rows, known_item 25,
  relation_graph_truth 20). Findings predating the doubling were measured on the old
  57,268-chunk / 2022–23 corpus — the pre-doubling sets are kept under
  `evals/datasets/_backup_predouble/`.
