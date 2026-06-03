"""§1.3 Chunking eval (slow, deterministic) — chunker invariants on a real PDF.

Guards the chunker against silent regressions that would degrade every downstream
stage: empty chunks, lost page metadata, or grossly oversized chunks (which blow
the embedding token budget). Parses a real local ΦΕΚ with docling, so it's slow
and needs docling's models cached — gated behind RUN_SLOW_EVALS so it never drags
the fast PR tier.

    RUN_SLOW_EVALS=1 pytest evals/ingestion -s
    RUN_SLOW_EVALS=1 CHUNK_PDF=downloads/2022/20220100175.pdf pytest evals/ingestion -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import settings

RUN = os.environ.get("RUN_SLOW_EVALS") == "1"

# Contextualize() prepends heading context, so a chunk's embedded text can exceed
# the raw token budget; allow generous headroom and only flag gross violations.
TOKEN_BUDGET_TOLERANCE = 1.5
MIN_PAGE_METADATA_RATE = 0.5


def _pick_pdf_with_text():
    """Return (pdf_path, chunks) for the first local PDF yielding >=3 chunks, so a
    single scanned/text-less PDF doesn't make the test flaky."""
    from src.ingestion.chunker import chunk_document
    from src.pdf.docling_loader import load_pdf

    env = os.environ.get("CHUNK_PDF")
    candidates = [Path(env)] if env else sorted(
        (settings.downloads_dir / "2022").glob("*.pdf"),
        key=lambda p: p.stat().st_size,
    )[:5]
    for pdf in candidates:
        if not pdf.is_file():
            continue
        chunks = [c for c in chunk_document(load_pdf(pdf)) if (c["text"] or "").strip()]
        if len(chunks) >= 3:
            return pdf, chunks
    return None, []


@pytest.mark.skipif(not RUN, reason="slow eval (docling parse); set RUN_SLOW_EVALS=1")
def test_chunking_invariants(capsys):
    import tiktoken

    pdf, chunks = _pick_pdf_with_text()
    if not chunks:
        pytest.skip("no local PDF with an extractable text layer found")

    try:
        enc = tiktoken.encoding_for_model(settings.openai_embedding_model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    # 1. No empty chunks.
    assert all((c["text"] or "").strip() for c in chunks)

    # 2. Token budget (gross-violation guard, with headroom for contextualization).
    tok_counts = [len(enc.encode(c["text"], disallowed_special=())) for c in chunks]
    ceiling = int(settings.chunk_tokens * TOKEN_BUDGET_TOLERANCE)
    over = [t for t in tok_counts if t > ceiling]

    # 3. Page metadata present on most chunks (needed for citations "σ. N").
    with_pages = sum(1 for c in chunks if c.get("pages"))
    page_rate = with_pages / len(chunks)

    report = {
        "pdf": pdf.name,
        "chunks": len(chunks),
        "max_tokens": max(tok_counts),
        "mean_tokens": round(sum(tok_counts) / len(tok_counts), 1),
        "over_budget": len(over),
        "page_metadata_rate": round(page_rate, 3),
    }
    with capsys.disabled():
        print("\n[chunking]", report)

    assert not over, f"{len(over)} chunks exceed {ceiling} tokens (budget {settings.chunk_tokens}): {over}"
    assert page_rate >= MIN_PAGE_METADATA_RATE, report
