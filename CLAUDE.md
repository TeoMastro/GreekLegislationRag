# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Operator procedures (load a year, verify the graph, run years in parallel) live
in **`README.md` → "Runbook: loading a year end-to-end"** — follow that for data
operations rather than reconstructing the steps. This file covers what's needed
to edit the code and run the project safely.

## What this is

Multi-agent RAG over Greek ΦΕΚ legislation. PDFs under `downloads/<year>/` are
OCR'd, chunked, embedded (OpenAI `text-embedding-3-small`, 1536d), and stored in
Supabase (pgvector + accent-folded FTS). A LangGraph multi-agent pipeline answers
questions in Greek with hybrid retrieval, and a citation graph (`law_nodes` +
`law_citations`) answers relational questions. All commands run via
`python -m src.main` (Click CLI in `src/main.py`).

## Commands

```bash
# Setup
python -m venv venv && venv\Scripts\activate    # Windows; use venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                            # then fill in secrets (see Config)
# Supabase: run the sql/ migrations in order in the SQL editor —
# 001_init → 002_citations → 003_hybrid_fts_or → 004_greek_fts_config.
# Order matters (002 has an FK to documents(id); 004 switches the FTS config).

# Tests — default-safe deterministic suite (no API keys); this is the PR tier.
pytest
pytest evals/component                          # same suite, explicit
pytest evals/component/test_normalize.py        # a single file
pytest evals/component/test_normalize.py::test_name   # a single test
pytest -m online                                # opt-in: hits live OpenAI/Supabase, costs money

# Run
python -m src.main ingest --year 2024           # OCR+embed+upsert one year (resumable)
python -m src.main query "..." --interactive    # ask the RAG agent
python -m src.main stats                         # chunk count, distinct PDFs, per-year breakdown
python -m src.main diagnose 20240100207.pdf     # extraction diagnostics, no DB write
python -m src.main sync-law-nodes                # populate law_nodes from listing files
python -m src.main extract-citations --year 2024 # backfill citation edges
```

There is **no separate lint/build step** — it's a plain Python package run as a
module. `pytest` is the gate.

## Architecture

- **Ingest** (`src/ingestion/ingest.py`): per PDF → OCR (`ocr_with_fallback`) →
  chunk → LLM metadata enrich → embed → `delete_by_source` then `insert_chunks`
  → inline citation extraction. Resumable via the `existing_sources()` skip set.
- **Query** (`src/rag/graph.py`, LangGraph): RewriterAgent (make follow-ups
  standalone) → ChunkAgent (hybrid cosine+FTS) ∥ ListingAgent (structured
  kind/number/year intent → scoped retrieval) ∥ RelationAgent (relational intent
  → queries the citation graph directly) → CombinerAgent (fuse, dedupe, `[n]`
  citations, validate against sources). Agents live in `src/rag/agents/`.
- **Citation graph** (`src/citations/`): `extractor.py` runs a regex pre-filter
  (`Ν. NNNN/YYYY`, `Π.Δ.`, `Π.Ν.Π.`, `Υ.Α.`) then a gpt-mini relation classifier
  over a closed vocabulary (τροποποιεί / καταργεί / αντικαθιστά / προστίθεται /
  αναφέρεται). `normalize.py` builds canonical keys; `backfill.py` is the CLI
  orchestration. This is what reaches relational candidates that top-K hybrid
  retrieval truncates away.
- **Storage** (`src/retrieval/store.py`): **all** Supabase reads/writes funnel
  through here — add new DB access here, not inline in callers. Schema lives in
  `sql/` (documents + `match_documents*` RPCs, law_nodes/law_citations + traversal
  RPCs, hybrid-FTS and Greek-FTS-config migrations). Reads paginate
  (`range(offset, offset+page-1)`, stop when a page is short); writes batch.
- **Hybrid retrieval** (`store.py` `_build_fts_query`): the lexical leg OR's
  accent-folded, stopword-filtered terms (capped at 6) — NOT AND'd, or a whole
  question becomes a conjunction no chunk satisfies (the leg was silently
  pure-semantic for a long time; see `evals/FINDINGS.md` #3). The FTS column uses
  Postgres' `greek` Snowball config, which stems but carries **no stopword
  dictionary** — the client-side `_FTS_STOPWORDS` set is the *only* stopword
  mechanism and an unfiltered ubiquitous term (e.g. `άρθρο`) trips
  `statement_timeout`. Don't remove the stopword filter or the term cap.
- **Config** (`src/config.py`): every tunable (models, OCR engine, chunk size,
  RRF weights, thresholds) is a pydantic-settings field with a default; `.env`
  overrides only what an environment needs.

## Invariants — easy to break, please preserve

- **OCR has no quality gate.** Ingest always OCRs (Mistral primary → Tesseract
  fallback). A previous text-layer "quality gate" produced garbled chunks — do
  not reintroduce trusting an embedded text layer. Files over
  `mistral_max_pdf_mb` (50) skip Mistral and go straight to Tesseract.
- **Ordering: `sync-law-nodes` before `ingest`/`extract-citations`.** Inline
  citation extraction skips any document whose source `law_node` doesn't exist
  yet (logs `Run sync-law-nodes first`), ingesting chunks without edges.
- **Parallel-run safety rests on idempotent upserts.** Running different years
  concurrently is safe only because `upsert_law_node` (on `canonical_key`),
  `insert_citations`, and `mark_chunks_processed` all use conflict-safe upserts /
  `ignore_duplicates`. Keep any new graph write idempotent.
- **Resumability.** `ingest` skips PDFs already in `documents`; `extract-citations`
  skips already-processed chunks. Preserve the skip-set checks when editing.
- **`delete_by_source` before `insert_chunks`** in `process_pdf` stops an
  interrupted re-run from stacking duplicate chunks. Keep that order.
- **RRF weights are eval-driven** (semantic 2× lexical, `1/2/50`). Read
  `evals/FINDINGS.md` before changing retrieval/ranking knobs in `config.py`.
- **GPT-5-family models reject `max_tokens`.** Pass the output cap as
  `model_kwargs={"max_completion_tokens": ...}` instead (see
  `combiner_agent.py:150`). Any new GPT-5 LLM call must follow this or it errors.
- **The compiled graph is a process-global singleton** (`graph.py` `_compiled_graph`),
  so the checkpointer is chosen once at first use — switching memory↔Postgres
  needs a process restart, not just an env change.

## Config / environment

`.env` (pydantic-settings, case-insensitive). Required: `OPENAI_API_KEY`,
`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`. For the default Mistral OCR:
`MISTRAL_API_KEY` (omit only if `OCR_ENGINE=tesseract`). Optional:
`CHECKPOINTER_DSN` (LangGraph multi-turn memory via PostgresSaver; falls back to
in-memory `MemorySaver`, lost on restart), `MISTRAL_MAX_PDF_MB`, `ENABLE_OCR`. Env vars override `config.py` defaults — when
runtime behavior surprises you (e.g. which OCR engine ran), check `.env` first.

The **Tesseract fallback path** needs Tesseract + Greek language data (`ell`) +
Ghostscript on `PATH` (ocrmypdf depends on Ghostscript). Docling downloads
layout models on first run (cached under `~/.cache/docling/`).

## Notes

- Greek text throughout — keep UTF-8 (`main.py` reconfigures stdio to utf-8).
- A harmless `ResourceTracker ... '_thread.RLock'` traceback on shutdown comes
  from the `multiprocess` dependency and does not indicate failure.
