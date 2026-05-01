# Greek Legislation RAG

Multi-agent Retrieval-Augmented Generation over Greek ΦΕΚ documents.
Parses PDFs with **Docling** (with **ocrmypdf + Tesseract** fallback for
scanned FEKs), embeds with **OpenAI** (`text-embedding-3-small`), stores
vectors in **Supabase pgvector**, and answers questions in Greek with
**GPT-5.x** using hybrid (semantic + full-text) retrieval.

---

## Architecture

```
PDF (downloads/YYYY/*.pdf)
  └─► Docling DocumentConverter   (layout + tables; do_ocr=False)
        └─► assess_text_quality   (chars / chars-per-page / page coverage)
              ├─ pass → Docling HybridChunker
              └─ fail → ocrmypdf + Tesseract (ell+eng), cached to downloads/.ocr/
                       └─► re-extract → Docling HybridChunker
                             └─► OpenAI embeddings (text-embedding-3-small, 1536d)
                                   └─► Supabase: documents
                                       (vector + accent-folded tsvector + jsonb)

Query (multi-agent LangGraph)
  └─► RewriterAgent  — make follow-ups standalone, embed once
        ├─► ChunkAgent    — hybrid retrieval (cosine + Greek-FTS via f_unaccent)
        └─► ListingAgent  — structured intent (kind, number, year, FEK title) →
                            scoped per-PDF hybrid retrieval
              └─► CombinerAgent — fuse, dedupe, format with [n] citations,
                                  validate citations against retrieved sources
```

Per-document metadata is built from two sources:
1. **Filename + folder** — `year`, `fek_series_code`, `sequence`, `source`.
2. **LLM enrichment** (optional, `gpt-4o-mini`) — `title`, `fek_number`, `publication_date`, `doc_type`, `authority`, `subject`.

---

## Setup

### 1. Supabase

In the Supabase SQL editor, paste and run [`sql/001_init.sql`](sql/001_init.sql).
This single bootstrap creates everything the app needs:
- `vector` and `unaccent` extensions
- `f_unaccent` IMMUTABLE wrapper (so the FTS column / index can use it)
- `documents` table (`content`, `metadata jsonb`, `embedding vector(1536)`,
  accent-folded `fts tsvector`)
- HNSW index on the embedding, GIN on `fts` and `metadata`, btree on
  `(metadata->>'source')` (used by `delete_by_source` during `--force` ingest)
- two RPC functions: `match_documents` (pure semantic) and
  `match_documents_hybrid` (RRF, accent-folded on both indexed text and query)
- `alter role service_role set statement_timeout = '60s'` — **required on
  free tier**, where the default 8s per-statement cap cancels bulk inserts of
  OCR'd PDFs while the HNSW index updates

### 2. Environment

```bash
cp .env.example .env
```

`.env` holds **secrets and per-environment values only**. All tunables
(model names, chunk size, retrieval weights, OCR thresholds, etc.) live in
[`src/config.py`](src/config.py) as defaults; override them in `.env` only
if a specific environment needs to.

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | sk-... |
| `SUPABASE_URL` | yes | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | yes | service-role key (writes) |
| `SUPABASE_ANON_KEY` | no | optional read-only key |
| `CHECKPOINTER_DSN` | no | Postgres DSN for LangGraph multi-turn memory; falls back to in-memory |

### 3. Python

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Docling will download its layout models on first run (a few hundred MB,
cached under `~/.cache/docling/`).

### 4. OCR for scanned PDFs (recommended for the FEK corpus)

Older FEK documents (and many gazette-signed PDFs) are **scans with only a
form-field text layer** — docling extracts ~100 chars from a 40-page document
and the chunk would be useless. The ingest pipeline detects this via three
quality thresholds (`min_text_chars`, `min_chars_per_page`, `min_page_coverage`
in `config.py`) and falls back to **`ocrmypdf` + Tesseract** with `ell+eng`
language data. The OCR'd PDF is cached at `downloads/.ocr/<rel-path>.pdf`,
so re-runs are free.

Native text-layer PDFs that pass the thresholds skip OCR entirely — there's
no speed cost for born-digital documents.

#### Windows install

You need three things on `PATH`: **Tesseract**, the **Greek language data
files**, and **Ghostscript** (ocrmypdf needs Ghostscript for PDF
rasterization). All free.

```powershell
# 1. Tesseract via winget (UAC prompt)
winget install --id UB-Mannheim.TesseractOCR --exact --silent `
    --accept-package-agreements --accept-source-agreements

# 2. Greek language data (Modern + Ancient).
#    `ell` is required (Modern Greek). `grc` is only needed to silence
#    Tesseract's OSD step that requests it for orientation detection.
$tessdata = "C:\Program Files\Tesseract-OCR\tessdata"
Invoke-WebRequest `
    -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/ell.traineddata" `
    -OutFile "$env:TEMP\ell.traineddata"
Invoke-WebRequest `
    -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/grc.traineddata" `
    -OutFile "$env:TEMP\grc.traineddata"
# Elevated copy (UAC prompt) into Program Files\Tesseract-OCR\tessdata\
Start-Process powershell -Verb RunAs -Wait -ArgumentList `
    "-NoProfile","-Command", @"
Copy-Item '$env:TEMP\ell.traineddata' '$tessdata\ell.traineddata' -Force
Copy-Item '$env:TEMP\grc.traineddata' '$tessdata\grc.traineddata' -Force
"@

# 3. Ghostscript — not in winget; install from https://www.ghostscript.com/releases/gsdnld.html
#    (AGPL 64-bit). Default path: C:\Program Files\gs\gs<version>\bin\gswin64c.exe

# 4. Add both to user PATH (adjust the gs version folder to whatever you installed)
[Environment]::SetEnvironmentVariable(
    "PATH",
    "C:\Program Files\Tesseract-OCR;C:\Program Files\gs\gs10.07.0\bin;" +
        [Environment]::GetEnvironmentVariable("PATH","User"),
    "User"
)
```

**Open a new PowerShell window** (PATH is read at shell startup), then verify:

```powershell
tesseract --list-langs   # should include ell, eng, grc, osd
gswin64c --version       # should print a version like 10.07.0
```

#### Sanity check on a single PDF

The `diagnose` subcommand runs the full extraction pipeline on one file and
prints metrics for both the native pass and the OCR pass — without writing
to the DB. Use this to confirm OCR works on your corpus before backfilling:

```bash
python -m src.main diagnose 20220100089.pdf
# (resolves filename anywhere under downloads/, skipping the .ocr cache)
```

Look for `chars/page` to jump from <300 (native, scan) to ~2000+ (OCR), and
`status: PASS` on the After-OCR table.

#### Linux / macOS

```bash
# Debian/Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-ell tesseract-ocr-grc ghostscript

# macOS (Homebrew)
brew install tesseract tesseract-lang ghostscript
```

Then `tesseract --list-langs` should include `ell`. No PATH tweaks needed.

### 5. PDFs

> **Run the scraper first.** This repo expects the `downloads/` folder to
> already contain ΦΕΚ PDFs. It does **not** fetch them itself — ingest will
> simply report "No PDFs found." on an empty tree.

Populate `downloads/` by cloning and running the companion scraper
[**GreekLegislationScrapper**](https://github.com/supernlogn/GreekLegislationScrapper)
**before** the first `ingest` run. The scraper downloads ΦΕΚ documents from
the official Greek government gazette and writes them into year-keyed folders
alongside per-year `listing-items-YYYY.md` manifests (used by `ListingAgent`
for structured-intent retrieval):

```
downloads/
├── 2022/  *.pdf
├── 2023/  *.pdf
├── 2024/  *.pdf
├── 2025/  *.pdf
├── 2026/  *.pdf
└── listing-items-YYYY.md   # one per year, parsed by ListingAgent
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

### Diagnose

Run the extraction pipeline on a single PDF without writing to the DB. Prints
character counts, page coverage, Greek vs. Latin char ratio, and the first 600
chars of extracted text. Runs both the native pass and the OCR fallback so you
can compare.

```bash
python -m src.main diagnose 20220100089.pdf
python -m src.main diagnose 20220100089.pdf --no-ocr   # native only
```

### Stats / reset

```bash
python -m src.main stats   # chunk count, distinct PDFs, breakdown by year
python -m src.main reset   # DESTRUCTIVE: truncates the documents table
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