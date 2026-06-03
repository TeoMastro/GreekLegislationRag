# Evals

Automated evaluation suite for the Greek Legislation RAG system. Organized in
**tiers by cost** so the cheap deterministic checks gate every change and the
expensive LLM/scale checks run on demand.

## Tiers

| Tier | Where | Cost | When |
|---|---|---|---|
| **Component** (deterministic) | `evals/component/` | free, no network | every PR (`pytest`) |
| **Retrieval** (recall@k) | `evals/retrieval/` | OpenAI + Supabase | nightly / on demand |
| **Routing/intent** (relation + listing) | `evals/routing/` | OpenAI | nightly / on demand |
| **E2E answer** (faithfulness, abstention) | `evals/e2e/` | OpenAI + Supabase | nightly / pre-release |
| **Scale** (latency, filtered recall, RAM) | `evals/scale/` | OpenAI + Supabase | weekly / on demand |
| **Relational graph** (routing → edges) | `evals/graph/` | OpenAI + Supabase | nightly / on demand |
| **Chunking** (invariants) | `evals/ingestion/` | local PDF + docling, no API | on demand (`RUN_SLOW_EVALS`) |

The routing layer is split: the pure intent post-processing (focal-key
resolution, structured filters) gates for free in `component/test_routing_logic.py`;
the LLM intent *extraction* is scored online in `routing/`.

`pytest.ini` scopes default collection to `evals/component`, so plain `pytest`
runs only the free tier. Online tiers are opt-in.

## Running

```bash
pip install -r requirements-dev.txt

# PR tier — deterministic, no credentials needed:
pytest                      # or: pytest evals/component
pytest evals/component -s   # -s prints the regex per-format recall breakdown

# Retrieval tier (needs a populated corpus + live keys in .env):
python -m evals.retrieval.synthetic_gen --n 200 --per-year 40   # build gold set
RUN_ONLINE_EVALS=1 pytest evals/retrieval -s

# Routing/intent tier (needs live OpenAI; no corpus required):
RUN_ONLINE_EVALS=1 pytest evals/routing -s

# Known-item tier (ListingAgent: "Ν. 5263" -> correct ΦΕΚ; needs corpus + manifests):
python -m evals.retrieval.gen_known_item --n 15
RUN_ONLINE_EVALS=1 pytest evals/retrieval/test_known_item.py -s

# E2E answer tier (runs the full pipeline; needs OpenAI + corpus). Slow.
RUN_ONLINE_EVALS=1 pytest evals/e2e -s
RUN_ONLINE_EVALS=1 E2E_N=2 pytest evals/e2e/test_faithfulness.py -s   # quick smoke

# Scale probe (read-only; latency + filtered recall + RAM projection):
python -m evals.scale.scale_probe --k 50 --filtered-sample 10

# Relational-graph tier (needs OpenAI + a populated citation graph):
python -m evals.graph.gen_relation_truth --n 15      # build truth from the graph
RUN_ONLINE_EVALS=1 pytest evals/graph -s

# Chunking invariants (slow; parses a local PDF with docling, no API):
RUN_SLOW_EVALS=1 pytest evals/ingestion -s
```

The synthetic gold set is generated with a **quality gate** (`synthetic_gen.py`
runs a second LLM pass that discards garbled / non-self-contained / unanswerable
questions); use `--oversample` to keep enough candidates to reach `--n` after
rejections, or `--no-validate` to skip the gate.

The component tier needs no real secrets: the root `conftest.py` injects
placeholder OpenAI/Supabase values so `src.config.settings` constructs, and these
tests never hit the network.

## Layout

```
evals/
  _util.py                 # dataset loader + report writer (offline)
  datasets/                # versioned, FROZEN golden data (jsonl)
    normalize_golden.jsonl     # §1.1 canonical-key table
    regex_candidates.jsonl     # §1.2 labeled chunks for the citation regex
    synthetic_queries.jsonl    # §2.1 generated; commit it to freeze the gold set
    relation_intent.jsonl      # §4.1 relational-routing gold set
    listing_intent.jsonl       # §4.2 structured-filter gold set
    relation_graph_truth.jsonl # §2.3 focal laws + true edges (generated)
    known_item.jsonl           # §2.2 "Ν. N" -> expected ΦΕΚ (generated)
  component/               # §1 + §4(pure) deterministic evals (pytest)
    test_normalize.py          # canonicalize_kind / canonical_key — 100% exact
    test_regex_candidates.py   # regex recall (the citation graph's ceiling)
    test_citation_format.py    # _strip_invalid_citations contract
    test_listing_parser.py     # listing-table parser vs scraper drift
    test_routing_logic.py      # focal-key resolution + hard structured filters
    test_text_quality.py       # §6 extraction-quality gate + garbled-text blind spot
    fixtures/
    unanswerable.jsonl         # §5.2 out-of-corpus queries for abstention
  retrieval/               # §2.1 synthetic gold-set generator + recall harness
                           # §2.4 ablation.py + embed_ablation.py (tuning scripts)
  routing/                 # §4.1/§4.2 online intent-extraction harnesses
  e2e/                     # §5 faithfulness + abstention (judges.py + tests)
  scale/                   # §3 scale_probe.py — latency / filtered recall / RAM
  graph/                   # §2.3 relational routing→edges (gen + test)
  ingestion/               # §1.3 chunking invariants (slow, RUN_SLOW_EVALS)
  report/                  # JSON metric snapshots (committed for trend tracking)
```

## Datasets are golden — keep them frozen

Edit a dataset only to **add** coverage or **fix a wrong label**, never to make a
failing test pass. Each row is a contract. `regex_candidates.jsonl` rows tagged
`known_gap` are shapes the regex provably misses today; they're tracked (and
counted in `recall_overall`) but don't gate CI. Remove the flag when the gap is
closed.

## Findings

Problems, suggested fixes, and methodology notes surfaced by the evals live in
[`FINDINGS.md`](FINDINGS.md). Raw metric snapshots are in `report/`.
