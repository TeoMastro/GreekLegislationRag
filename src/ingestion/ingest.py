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


def _iter_pdfs(year: int | None) -> list[Path]:
    root = settings.downloads_dir
    if not root.exists():
        return []
    if year is not None:
        return sorted((root / str(year)).glob("*.pdf"))
    return sorted(root.glob("**/*.pdf"))


def _resolve_file(file: str) -> Path | None:
    p = Path(file)
    if p.is_file():
        return p
    root = settings.downloads_dir
    matches = list(root.glob(f"**/{file}"))
    if not matches and not file.lower().endswith(".pdf"):
        matches = list(root.glob(f"**/{file}.pdf"))
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


_MIN_TEXT_CHARS = 100


def process_pdf(pdf: Path, force: bool = False) -> int:
    doc = load_pdf(pdf)
    chunks = [c for c in chunk_document(doc) if c["text"] and c["text"].strip()]
    total_chars = sum(len(c["text"].strip()) for c in chunks)
    if not chunks or total_chars < _MIN_TEXT_CHARS:
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
