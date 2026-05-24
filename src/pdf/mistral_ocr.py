"""Mistral OCR engine — a cloud alternative to the local ocrmypdf+Tesseract path.

Unlike ocrmypdf (PDF in → searchable PDF out, re-extracted by Docling), Mistral
returns per-page markdown directly. We therefore build chunks here in the same
``{text, headings, pages}`` shape the Docling chunker emits, preserving the page
number on each chunk so the sources table and citation snippets stay accurate.

The raw per-page markdown is cached at ``downloads/.ocr/<rel>.pdf.mistral.json``
(keyed by source mtime) because every call is a paid, per-page API request.
"""

import base64
import json
from functools import lru_cache
from pathlib import Path

import httpx
import tiktoken

from src.config import settings

_OCR_ENDPOINT = "https://api.mistral.ai/v1/ocr"


def _cache_root() -> Path:
    return settings.downloads_dir / ".ocr"


def _cache_path(pdf: Path) -> Path:
    try:
        rel = pdf.resolve().relative_to(settings.downloads_dir.resolve())
    except ValueError:
        rel = Path(pdf.name)
    return _cache_root() / rel.parent / (rel.name + ".mistral.json")


@lru_cache(maxsize=1)
def _encoding():
    try:
        return tiktoken.encoding_for_model(settings.openai_embedding_model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _fetch_markdown(pdf: Path) -> list[str]:
    """Return per-page markdown for ``pdf``, hitting the API only on a cache miss."""
    out = _cache_path(pdf)
    if out.exists() and out.stat().st_mtime >= pdf.stat().st_mtime:
        return json.loads(out.read_text(encoding="utf-8"))

    if not settings.mistral_api_key:
        raise RuntimeError(
            "OCR_ENGINE=mistral but MISTRAL_API_KEY is not set (see .env.example)."
        )

    # The OCR endpoint is a single JSON POST, so we call it directly rather than
    # pull in the heavyweight mistralai SDK (opentelemetry et al.). httpx already
    # ships with the project via openai.
    b64 = base64.b64encode(pdf.read_bytes()).decode("utf-8")
    resp = httpx.post(
        _OCR_ENDPOINT,
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.mistral_ocr_model,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    pages = sorted(resp.json().get("pages", []), key=lambda p: p.get("index", 0))
    markdown = [p.get("markdown") or "" for p in pages]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(markdown, ensure_ascii=False), encoding="utf-8")
    return markdown


def _is_heading(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) == 1 and lines[0].lstrip().startswith("#")


def _split_page(markdown: str, page_no: int, max_tokens: int) -> list[dict]:
    """Token-bounded split of one page's markdown into chunks tagged with ``page_no``.

    Splits on blank lines, accumulating blocks until the next would exceed
    ``max_tokens``. The most recent markdown heading is carried onto each chunk.
    """
    enc = _encoding()
    chunks: list[dict] = []
    cur: list[str] = []
    cur_tokens = 0
    last_heading: str | None = None

    def flush() -> None:
        nonlocal cur, cur_tokens
        text = "\n\n".join(cur).strip()
        if text:
            chunks.append(
                {
                    "text": text,
                    "headings": [last_heading] if last_heading else [],
                    "pages": [page_no],
                }
            )
        cur = []
        cur_tokens = 0

    for raw in markdown.split("\n\n"):
        block = raw.strip()
        if not block:
            continue
        if _is_heading(block):
            last_heading = block.lstrip().lstrip("#").strip()

        n_tokens = len(enc.encode(block))
        if n_tokens > max_tokens:
            # A single block too large to fit — flush what we have, then hard-split it.
            flush()
            toks = enc.encode(block)
            for i in range(0, len(toks), max_tokens):
                piece = enc.decode(toks[i : i + max_tokens]).strip()
                if piece:
                    chunks.append(
                        {
                            "text": piece,
                            "headings": [last_heading] if last_heading else [],
                            "pages": [page_no],
                        }
                    )
            continue

        if cur and cur_tokens + n_tokens > max_tokens:
            flush()
        cur.append(block)
        cur_tokens += n_tokens

    flush()
    return chunks


def ocr_pdf_to_chunks(pdf: Path) -> tuple[list[dict], int]:
    """OCR ``pdf`` via Mistral and return ``(chunks, total_pages)``.

    Mirrors the return contract of ``_extract`` so it slots into the ingest
    pipeline exactly where the Tesseract path produces its re-extracted chunks.
    """
    pages = _fetch_markdown(pdf)
    max_tokens = settings.chunk_tokens
    chunks: list[dict] = []
    for i, md in enumerate(pages):
        chunks.extend(_split_page(md, i + 1, max_tokens))
    chunks = [c for c in chunks if c["text"].strip()]
    return chunks, len(pages)
