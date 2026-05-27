# Greek Legislation RAG

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENCE.md)

Multi-agent Retrieval-Augmented Generation over Greek ΦΕΚ documents.
OCRs every PDF — **Mistral OCR** via API by default, falling back to local
**ocrmypdf + Tesseract + Docling** when Mistral errors or can't take the file
(e.g. very large PDFs) — embeds with **OpenAI** (`text-embedding-3-small`),
stores vectors in **Supabase pgvector**, and answers questions in Greek with
**GPT-5.x** using hybrid (semantic + full-text) retrieval.

---

## Architecture

```
PDF (downloads/YYYY/*.pdf)
  └─► OCR every PDF — primary engine → fallback (cached to downloads/.ocr/):
        ├─ mistral   : Mistral OCR API → per-page markdown chunked in place
        │              (skipped when PDF > MISTRAL_MAX_PDF_MB)
        └─ tesseract : ocrmypdf+Tesseract (ell+eng) → Docling re-extract → HybridChunker
              └─► OpenAI embeddings (text-embedding-3-small, 1536d)
                    └─► Supabase: documents
                        (vector + accent-folded tsvector + jsonb)

Query (multi-agent LangGraph)
  └─► RewriterAgent  — make follow-ups standalone, embed once
        ├─► ChunkAgent     — hybrid retrieval (cosine + Greek-FTS via f_unaccent)
        ├─► ListingAgent   — structured intent (kind, number, year, FEK title) →
        │                    scoped per-PDF hybrid retrieval
        └─► RelationAgent  — detects relational intent ("which laws does X
                             amend / which laws affected X"), queries the
                             citation graph (law_nodes + law_citations) directly
              └─► CombinerAgent — three-stream fuse, dedupe, format with [n]
                                  citations, validate citations against sources
```

The **citation graph** (`law_nodes`, `law_citations`) is the answer to
relational questions that pure vector / hybrid retrieval can't reach — top-K
truncation hides most of the candidates. Edges are extracted at ingest time
by a regex pre-filter (matches `Ν. NNNN/YYYY`, `Π.Δ.`, `Π.Ν.Π.`, `Υ.Α.`)
followed by a gpt-5.4-mini relation classifier (closed vocabulary:
*τροποποιεί / καταργεί / αντικαθιστά / προστίθεται / αναφέρεται*). Chunks
without any candidate span never reach the LLM.

Per-document metadata is built from two sources:
1. **Filename + folder** — `year`, `fek_series_code`, `sequence`, `source`.
2. **LLM enrichment** (optional, `gpt-4o-mini`) — `title`, `fek_number`, `publication_date`, `doc_type`, `authority`, `subject`.

---

## Setup

### 1. Supabase

In the Supabase SQL editor, paste and run [`sql/001_init.sql`](sql/001_init.sql),
then [`sql/002_citations.sql`](sql/002_citations.sql). Run 001 first — 002
references `documents(id)` from a foreign key.

`002_citations.sql` adds the citation graph: `law_nodes` (one row per
canonical law, keyed by `Ν.NNNN/YYYY`-style `canonical_key`), `law_citations`
(one row per extracted edge — source chunk, target law, relation, article,
snippet, confidence), and two RPCs `citations_from_law` / `citations_to_law`
for outgoing / incoming traversal.

`001_init.sql` creates everything the chunk store needs:
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
| `OCR_ENGINE` | no | primary OCR engine: `mistral` (default, cloud) or `tesseract` (local); the other is the automatic fallback |
| `MISTRAL_API_KEY` | no | required for the default `mistral` engine; omit only if you set `OCR_ENGINE=tesseract` |
| `MISTRAL_MAX_PDF_MB` | no | PDFs larger than this (default `50`) skip Mistral and go straight to Tesseract |
| `ENABLE_OCR` | no | `true` (default) OCRs every PDF; `false` uses Docling's native text layer only |

### 3. Python

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Docling will download its layout models on first run (a few hundred MB,
cached under `~/.cache/docling/`).

### 4. OCR (every PDF is OCR'd)

Every PDF is OCR'd — too many FEKs have broken or glyph-mangled text layers to
trust native extraction. The default engine is **Mistral OCR** (cloud); when it
errors, returns nothing, or the PDF exceeds `MISTRAL_MAX_PDF_MB`, ingest falls
back to **`ocrmypdf` + Tesseract** with `ell+eng` language data. Both outputs
are cached under `downloads/.ocr/` (searchable `*.pdf` for Tesseract,
`*.pdf.mistral.json` for Mistral), so re-runs are free. The engine that actually
produced each chunk is recorded in `metadata.ocr_engine`.

You still need the Tesseract toolchain (below) installed for the fallback path.
To OCR with Tesseract only, set `OCR_ENGINE=tesseract`; to skip OCR entirely and
use Docling's native text layer, set `ENABLE_OCR=false`.

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

#### Mistral OCR (default engine)

[**Mistral OCR**](https://docs.mistral.ai/capabilities/OCR/basic_ocr/) is the
default OCR engine — often stronger on Greek than Tesseract, and it needs **no
system dependencies** (no Tesseract, no Ghostscript) and **no extra Python
packages** (it's a plain REST call over the `httpx` that already ships with the
project). It's a paid, per-page API call. Set your key in `.env`:

```bash
OCR_ENGINE=mistral        # default
MISTRAL_API_KEY=...
```

Where ocrmypdf produces a searchable PDF that Docling re-parses, Mistral returns
per-page markdown directly, which is chunked in place with the page number
preserved on every chunk. Raw markdown is cached at
`downloads/.ocr/<rel>.pdf.mistral.json` (keyed by source mtime), so re-runs and
`diagnose` don't re-bill. PDFs larger than `MISTRAL_MAX_PDF_MB` (default 50)
skip Mistral's inline upload and fall back to Tesseract automatically. Each
chunk carries `metadata.ocr_engine` recording which engine produced it.

`diagnose` honours `OCR_ENGINE`, so you can compare cost/quality on one file
before backfilling:

```bash
OCR_ENGINE=mistral python -m src.main diagnose 20220100089.pdf
```

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

### Citation graph

The relational layer is two commands. New ingests run the extractor
automatically (gated by `settings.enable_citation_extraction`); these
commands populate it from an already-loaded corpus.

```bash
# 1. Populate law_nodes from downloads/listing-items-YYYY.md. Idempotent.
#    Run this first — extract-citations needs source PDFs resolved to nodes.
python -m src.main sync-law-nodes

# 2. Smoke test: extract from one year before paying for the full backfill.
python -m src.main extract-citations --year 2024

# 3. Full backfill once the smoke test looks right.
python -m src.main extract-citations

# 4. Debug a single PDF.
python -m src.main extract-citations --file 20240100207.pdf

# 5. Reprocess chunks that already have edges (e.g. after improving the prompt).
python -m src.main extract-citations --year 2024 --reprocess
```

The backfill is **resumable** — it queries `law_citations` for chunk ids
already processed and skips them. Citation rows dedup via a unique constraint
on `(source_chunk_id, target_law_id, relation, target_article)`, so duplicate
runs are safe but wasteful (use `--reprocess` only when intended).

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

For **relational** questions (*"Ποιους νόμους τροποποιεί ο Ν. 5167/2024;"*,
*"Ποιοι νόμοι κατήργησαν τον ν. 4001/2011;"*) `RelationAgent` runs in
parallel with the chunk and listing agents. It:

1. Asks gpt-5.4-mini to classify intent — `is_relational`, `direction`
   (`outgoing` / `incoming` / `both`), `source_law` `(kind, number, year)`,
   and an optional `relations` filter.
2. Resolves the focal law to a canonical key (`Ν.5167/2024`) and calls
   `citations_from_law` / `citations_to_law` RPCs filtered by
   `settings.citation_min_confidence`.
3. Hydrates each citation row into a `Document` by re-fetching its source
   chunk, then attaches a `_relation_match` metadata block.

The combiner gives graph-derived hits a rank boost weighted by classifier
confidence, so the chunks that *actually answer* the relation question rise
to the top of the fused context — instead of being shadowed by lexically
similar but irrelevant chunks. For non-relational queries the agent returns
empty results and the original chunk + listing pipeline takes over unchanged.

---

## License

This project is licensed under the **GNU General Public License v3.0** (or
any later version). See [`LICENCE.md`](LICENCE.md) for the full text.

GPL v3 is a **copyleft** license: you can use, study, modify, and
redistribute this software, and you may charge for copies or services, **but
any derivative work you distribute must also be released under GPL v3** and
its complete corresponding source code must be made available to recipients.
Combining this code with proprietary software in a single conveyed work is
not permitted.

When using this project, please preserve the copyright notice and license
reference in source files. Modified versions must carry prominent notices
stating that they have been changed and the date of the change (see GPL v3
§5).

> *Note: GPL v3 does not prohibit commercial use — it only requires that
> derivatives remain free software. If you redistribute or host a modified
> version, you must publish the source under the same terms.*

---