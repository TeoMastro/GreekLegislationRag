# Greek Legislation RAG

Single-agent Retrieval-Augmented Generation over Greek ΦΕΚ documents.
Parses PDFs with **Docling**, embeds with **OpenAI** (`text-embedding-3-small`),
stores vectors in **Supabase pgvector**, and answers questions with **GPT-5.1**
using hybrid (semantic + full-text) retrieval.

---

## Architecture

```
PDF (downloads/YYYY/*.pdf)
  └─► Docling DocumentConverter   (layout + tables + OCR fallback)
        └─► Docling HybridChunker (heading-aware, token-capped)
              └─► OpenAI embeddings (text-embedding-3-small, 1536d)
                    └─► Supabase: documents (vector + tsvector + jsonb metadata)

Query
  └─► OpenAI embedding
        └─► Supabase RPC: match_documents_hybrid (RRF: cosine + full-text)
              └─► GPT-5.1 with retrieved context  →  cited answer
```

Per-document metadata is built from two sources:
1. **Filename + folder** — `year`, `fek_series_code`, `sequence`, `source`.
2. **LLM enrichment** (optional, `gpt-4o-mini`) — `title`, `fek_number`, `publication_date`, `doc_type`, `authority`, `subject`.

---

## Setup

### 1. Supabase

In the Supabase SQL editor, paste and run [`sql/001_init.sql`](sql/001_init.sql).
This creates:
- the `documents` table (`content`, `metadata jsonb`, `embedding vector(1536)`, `fts tsvector`)
- HNSW index on the embedding, GIN indexes on `fts` and `metadata`
- two RPC functions: `match_documents` (pure semantic) and `match_documents_hybrid` (RRF)

### 2. Environment

```bash
cp .env.example .env
```

Fill in:

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | sk-... |
| `SUPABASE_URL` | yes | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | yes | service-role key (writes) |
| `OPENAI_CHAT_MODEL` | no | default `gpt-5.1` |
| `OPENAI_EMBEDDING_MODEL` | no | default `text-embedding-3-small` |
| `METADATA_LLM_MODEL` | no | default `gpt-4o-mini` |
| `ENABLE_LLM_METADATA` | no | `true` / `false` |
| `CHUNK_TOKENS` | no | default `512` |
| `TOP_K` | no | default `10` |
| `RRF_K` | no | default `50` |

### 3. Python

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Docling will download its layout / OCR models on first run (a few hundred MB,
cached under `~/.cache/docling/`).

> **Tesseract for scanned PDFs (optional).** Docling falls back to OCR only if
> text extraction yields nothing. Install Tesseract and the Greek language pack
> (`ell`) if your corpus contains scanned ΦΕΚ documents. For digital ΦΕΚs you
> can skip this.

### 4. PDFs

Place your PDFs in year-keyed folders:

```
downloads/
├── 2022/  *.pdf
├── 2023/  *.pdf
├── 2024/  *.pdf
├── 2025/  *.pdf
└── 2026/  *.pdf
```

Filenames follow the `YYYYDDDNNNN.pdf` convention so `year`, `fek_series_code`
(e.g. `100` = Α series), and `sequence` are extracted automatically.

---

## CLI

All commands run via `python -m src.main`.

### Ingest

```bash
# Full corpus (skips PDFs already in DB)
python -m src.main ingest

# One year only
python -m src.main ingest --year 2024

# First 3 PDFs of a year — handy smoke test
python -m src.main ingest --year 2026 --limit 3

# Re-ingest everything (ignore skip set)
python -m src.main ingest --force
```

The ingester is **resumable** — it queries Supabase for already-ingested
filenames and skips them. Failures are reported at the end without aborting
the rest.

### Query

```bash
# One-off question
python -m src.main query "Ποιες είναι οι αλλαγές στη φορολογία ακινήτων το 2024;"

# Top-K and year filter
python -m src.main query "..." -k 15 --year 2024

# Interactive REPL
python -m src.main query --interactive
```

The output shows the answer (with `[n]` citations) and a sources table with
filename, year, page, semantic similarity, and RRF rank.

### Stats / reset

```bash
python -m src.main stats   # chunk count, distinct PDFs, breakdown by year
python -m src.main reset   # DESTRUCTIVE: truncates the documents table
```

---

## Project layout

```
.
├── sql/001_init.sql          # Supabase bootstrap (run once)
├── requirements.txt
├── .env.example
├── downloads/                # your PDFs, organized by year
└── src/
    ├── config.py             # pydantic-settings (.env)
    ├── main.py               # Click CLI
    ├── pdf/docling_loader.py # DocumentConverter wrapper
    ├── ingestion/
    │   ├── chunker.py        # HybridChunker + tiktoken
    │   ├── metadata.py       # filename parser + LLM enrichment
    │   ├── embedder.py       # batched OpenAI embeddings (tenacity retry)
    │   └── ingest.py         # walk → parse → chunk → embed → upsert
    ├── retrieval/store.py    # Supabase client, hybrid RPC
    └── rag/agent.py          # embed → hybrid retrieve → generate
```

---

## How retrieval works

`match_documents_hybrid` does **Reciprocal Rank Fusion** of two ranked lists:

1. **Semantic** — `embedding <=> query_embedding` (pgvector cosine distance, HNSW).
2. **Full-text** — `ts_rank_cd(fts, websearch_to_tsquery('simple', query_text))`.

Each side gets `match_count * 2` candidates; the SQL fuses them with
`1 / (rrf_k + rank_ix)` and weights, sorts by the fused score, returns
`match_count`. Both `full_text_weight` and `semantic_weight` default to `1.0`.

The `simple` text-search config skips stemming/stopwords — fine for ΦΕΚ
numbers and other exact tokens. If recall on Greek keywords feels weak,
install Postgres `unaccent` and wrap `content` / queries with `unaccent(...)`.

---

## Cost notes

For 422 PDFs, rough one-time cost:
- **Embeddings** (`text-embedding-3-small`): a few cents — large docs but a small model.
- **Metadata enrichment** (`gpt-4o-mini`, optional, one call per PDF): ~\$0.10–\$0.40 total.
- **Docling**: free (local models).
- **Per query**: one embedding (cents per thousand) + one GPT-5.1 chat with
  ~10 chunks of context.

Set `ENABLE_LLM_METADATA=false` to skip the per-doc enrichment call entirely.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: docling_core.transforms.chunker.tokenizer.openai` | Docling version mismatch — `pip install -U docling docling-core`. |
| First ingest is slow | Docling is downloading layout/OCR models. Subsequent runs use the cache. |
| `match_documents_hybrid` not found | The SQL script wasn't run, or it ran on a different schema. Re-run in the right project. |
| Empty `pages` in metadata | Docling didn't produce per-item provenance for that chunk; the chunk still works for retrieval. |
| `permission denied for table documents` | You're using the anon key. Use the **service role** key for ingest. |
