from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import settings
from src.pdf.docling_loader import load_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.metadata import metadata_from_path, enrich_with_llm
from src.ingestion.embedder import embed_texts
from src.ingestion.quality import assess_text_quality
from src.retrieval.store import delete_by_source, existing_sources, insert_chunks


console = Console()


def _is_ocr_cache(p: Path) -> bool:
    return ".ocr" in p.parts


def _iter_pdfs(year: int | None) -> list[Path]:
    root = settings.downloads_dir
    if not root.exists():
        return []
    if year is not None:
        return sorted((root / str(year)).glob("*.pdf"))
    return sorted(p for p in root.glob("**/*.pdf") if not _is_ocr_cache(p))


def _resolve_file(file: str) -> Path | None:
    p = Path(file)
    if p.is_file():
        return p
    root = settings.downloads_dir
    matches = [m for m in root.glob(f"**/{file}") if not _is_ocr_cache(m)]
    if not matches and not file.lower().endswith(".pdf"):
        matches = [
            m for m in root.glob(f"**/{file}.pdf") if not _is_ocr_cache(m)
        ]
    if not matches:
        return None
    if len(matches) > 1:
        console.print(
            f"[red]Ambiguous file '{file}' — {len(matches)} matches:[/red]"
        )
        for m in matches:
            console.print(f"  - {m}")
        return None
    return matches[0]

def _write_skip_report(
    year: int | None,
    too_large: list[tuple[Path, int]],
    no_text: list[Path],
    failures: list[tuple[Path, str]],
) -> None:
    """Append this run's skipped/failed PDFs to downloads/.skipped.log.

    A durable record (the terminal summary scrolls away on long runs). Each run
    appends a timestamped block; nothing is written when nothing was skipped.
    """
    if not (too_large or no_text or failures):
        return
    from datetime import datetime

    log = settings.downloads_dir / ".skipped.log"
    scope = f"year={year}" if year is not None else "all years"
    lines = [f"# {datetime.now().isoformat(timespec='seconds')}  ({scope})"]
    for pdf, pages in too_large:
        lines.append(f"too_large\t{pdf.name}\t{pages} pages")
    for pdf in no_text:
        lines.append(f"no_text\t{pdf.name}\timage-only / no extractable text")
    for pdf, msg in failures:
        lines.append(f"failure\t{pdf.name}\t{msg}")
    lines.append("")
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        console.print(f"[dim]Skip/failure details appended to {log}[/dim]")
    except OSError as e:
        console.print(f"[yellow]Could not write skip log: {e}[/yellow]")


def _page_count(pdf: Path) -> int | None:
    """Cheap page count without OCR; None if the PDF can't be read."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf)).pages)
    except Exception:
        return None


def _extract(pdf: Path) -> tuple[list[dict], int]:
    doc = load_pdf(pdf)
    chunks = [c for c in chunk_document(doc) if c["text"] and c["text"].strip()]
    total_pages = len(getattr(doc, "pages", None) or {})
    return chunks, total_pages


def _ocr_mistral(pdf: Path) -> tuple[list[dict], int]:
    from src.pdf.mistral_ocr import ocr_pdf_to_chunks

    return ocr_pdf_to_chunks(pdf)


def _ocr_tesseract(pdf: Path) -> tuple[list[dict], int]:
    from src.pdf.ocr import ensure_ocr

    return _extract(ensure_ocr(pdf))


_OCR_ENGINES = {"mistral": _ocr_mistral, "tesseract": _ocr_tesseract}


def _too_big_for_mistral(pdf: Path) -> bool:
    try:
        return pdf.stat().st_size > settings.mistral_max_pdf_mb * 1024 * 1024
    except OSError:
        return False


def ocr_with_fallback(pdf: Path) -> tuple[list[dict], int, str | None]:
    """OCR a PDF via the primary engine, falling back to the other on failure.

    Order is ``settings.ocr_engine`` first (default ``"mistral"``), then the
    other engine. A PDF larger than ``settings.mistral_max_pdf_mb`` skips the
    Mistral inline upload (which the API would reject anyway) and goes straight
    to the Tesseract/ocrmypdf path. Returns ``(chunks, total_pages, engine)``;
    ``engine`` is ``None`` only when every engine yielded no usable text.
    """
    primary = settings.ocr_engine if settings.ocr_engine in _OCR_ENGINES else "mistral"
    fallback = "tesseract" if primary == "mistral" else "mistral"

    order: list[str] = []
    for engine in (primary, fallback):
        if engine == "mistral" and _too_big_for_mistral(pdf):
            console.print(
                f"[yellow]{pdf.name}: {pdf.stat().st_size / 1048576:.0f}MB over the "
                f"{settings.mistral_max_pdf_mb}MB Mistral limit — using Tesseract.[/yellow]"
            )
            continue
        order.append(engine)

    for engine in order:
        try:
            chunks, total_pages = _OCR_ENGINES[engine](pdf)
            chunks = [c for c in chunks if c["text"] and c["text"].strip()]
            if chunks:
                return chunks, total_pages, engine
            console.print(
                f"[yellow]{pdf.name}: {engine} produced no text — trying next engine.[/yellow]"
            )
        except Exception as e:
            console.print(
                f"[yellow]{pdf.name}: {engine} OCR failed ({e}) — trying next engine.[/yellow]"
            )
    return [], 0, None


def process_pdf(pdf: Path) -> int:
    if settings.enable_ocr:
        console.print(
            f"[dim]{pdf.name}: OCR ({settings.ocr_engine} → fallback)...[/dim]"
        )
        chunks, total_pages, engine = ocr_with_fallback(pdf)
    else:
        chunks, total_pages = _extract(pdf)
        engine = None

    if not chunks:
        console.print(f"[yellow]Skipping {pdf.name}: no text extracted[/yellow]")
        return 0

    ocr_used = engine is not None

    base_meta = metadata_from_path(pdf)
    full_text = "\n\n".join(c["text"] for c in chunks[:5])
    llm_meta = enrich_with_llm(full_text)
    doc_meta = {**llm_meta, **base_meta}

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    rows: list[dict] = []
    total = len(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append(
            {
                "content": chunk["text"],
                "embedding": emb,
                "metadata": {
                    **doc_meta,
                    "chunk_index": i,
                    "total_chunks": total,
                    "headings": chunk["headings"],
                    "pages": chunk["pages"],
                    "embedding_model": settings.openai_embedding_model,
                    "embedding_dim": settings.openai_embedding_dimension,
                    "ocr_used": ocr_used,
                    **({"ocr_engine": engine} if engine else {}),
                },
            }
        )
    # Always clear any existing rows for this source before inserting, so a
    # re-processed PDF can never stack a second copy on top of old rows
    # (interrupted or overlapping runs). Skipped sources never reach here, so
    # this is free on the normal resumable path.
    delete_by_source(pdf.name)
    inserted_ids = insert_chunks(rows)

    if settings.enable_citation_extraction and inserted_ids:
        try:
            from src.citations.backfill import extract_for_chunks

            extract_for_chunks(
                pdf.name,
                list(zip(inserted_ids, [c["text"] for c in chunks])),
            )
        except Exception as e:
            console.print(
                f"[yellow]{pdf.name}: citation extraction failed (chunks "
                f"still ingested): {e}[/yellow]"
            )

    return len(rows)


def ingest(
    year: int | None = None,
    limit: int | None = None,
    force: bool = False,
    file: str | None = None,
) -> None:
    if file is not None:
        resolved = _resolve_file(file)
        if resolved is None:
            console.print(f"[red]File not found:[/red] {file}")
            return
        pdfs = [resolved]
    else:
        pdfs = _iter_pdfs(year)
        if limit is not None:
            pdfs = pdfs[:limit]
    if not pdfs:
        console.print("[yellow]No PDFs found.[/yellow]")
        return

    skip = set() if force else existing_sources()
    pending = [p for p in pdfs if p.name not in skip]
    skipped = len(pdfs) - len(pending)

    console.print(
        f"[bold]Found {len(pdfs)} PDFs[/bold] "
        f"({len(pending)} to process, {skipped} already ingested)"
    )

    total_chunks = 0
    no_text: list[Path] = []
    too_large: list[tuple[Path, int]] = []
    failures: list[tuple[Path, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting", total=len(pending))
        for pdf in pending:
            progress.update(task, description=f"Ingesting {pdf.name}")
            pages = _page_count(pdf)
            if pages is not None and pages > settings.max_pdf_pages:
                console.print(
                    f"[yellow]Skipping {pdf.name}: {pages} pages "
                    f"(> {settings.max_pdf_pages} limit)[/yellow]"
                )
                too_large.append((pdf, pages))
                progress.advance(task)
                continue
            try:
                n = process_pdf(pdf)
                if n == 0:
                    no_text.append(pdf)
                else:
                    total_chunks += n
            except Exception as e:
                failures.append((pdf, str(e)))
            progress.advance(task)

    inserted_docs = len(pending) - len(failures) - len(no_text) - len(too_large)
    console.print(
        f"\n[green]Done.[/green] {total_chunks} chunks inserted from "
        f"{inserted_docs} PDFs."
    )
    if too_large:
        console.print(
            f"[yellow]{len(too_large)} skipped (over "
            f"{settings.max_pdf_pages}-page limit):[/yellow]"
        )
        for pdf, pages in too_large[:20]:
            console.print(f"  - {pdf.name} ({pages} pages)")
    if no_text:
        console.print(
            f"[yellow]{len(no_text)} skipped (no extractable text — image-only PDFs):[/yellow]"
        )
        for pdf in no_text[:20]:
            console.print(f"  - {pdf.name}")
    if failures:
        console.print(f"[red]{len(failures)} failures:[/red]")
        for pdf, msg in failures[:20]:
            console.print(f"  - {pdf.name}: {msg}")

    _write_skip_report(year, too_large, no_text, failures)
