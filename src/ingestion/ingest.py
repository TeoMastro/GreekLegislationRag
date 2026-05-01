from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.config import settings
from src.pdf.docling_loader import load_pdf
from src.ingestion.chunker import chunk_document
from src.ingestion.metadata import metadata_from_path, enrich_with_llm
from src.ingestion.embedder import embed_texts
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


def assess_text_quality(
    chunks: list[dict], total_pages: int
) -> tuple[bool, str]:
    if not chunks:
        return False, "no extractable text"
    total_chars = sum(len(c["text"].strip()) for c in chunks)
    if total_chars < settings.min_text_chars:
        return False, f"only {total_chars} chars total"
    if total_pages > 0:
        pages_with_text = len({p for c in chunks for p in (c.get("pages") or [])})
        chars_per_page = total_chars / total_pages
        coverage = pages_with_text / total_pages
        if chars_per_page < settings.min_chars_per_page:
            return False, (
                f"{chars_per_page:.0f} chars/page over {total_pages} pages "
                f"(likely scanned)"
            )
        if coverage < settings.min_page_coverage:
            return False, (
                f"only {pages_with_text}/{total_pages} pages have text "
                f"(likely scanned)"
            )
    return True, ""


def _extract(pdf: Path) -> tuple[list[dict], int]:
    doc = load_pdf(pdf)
    chunks = [c for c in chunk_document(doc) if c["text"] and c["text"].strip()]
    total_pages = len(getattr(doc, "pages", None) or {})
    return chunks, total_pages


def process_pdf(pdf: Path, force: bool = False) -> int:
    chunks, total_pages = _extract(pdf)
    ok, reason = assess_text_quality(chunks, total_pages)
    ocr_used = False

    if not ok and settings.enable_ocr:
        console.print(
            f"[yellow]{pdf.name}: {reason} — running OCR "
            f"(this can take a few minutes)...[/yellow]"
        )
        try:
            from src.pdf.ocr import ensure_ocr

            ocr_pdf = ensure_ocr(pdf)
        except Exception as e:
            console.print(f"[red]{pdf.name}: OCR failed: {e}[/red]")
            return 0
        chunks, total_pages = _extract(ocr_pdf)
        ok, reason = assess_text_quality(chunks, total_pages)
        ocr_used = True

    if not ok:
        console.print(f"[yellow]Skipping {pdf.name}: {reason}[/yellow]")
        return 0

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
                },
            }
        )
    if force:
        delete_by_source(pdf.name)
    insert_chunks(rows)
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
            try:
                n = process_pdf(pdf, force=force)
                if n == 0:
                    no_text.append(pdf)
                else:
                    total_chunks += n
            except Exception as e:
                failures.append((pdf, str(e)))
            progress.advance(task)

    inserted_docs = len(pending) - len(failures) - len(no_text)
    console.print(
        f"\n[green]Done.[/green] {total_chunks} chunks inserted from "
        f"{inserted_docs} PDFs."
    )
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
