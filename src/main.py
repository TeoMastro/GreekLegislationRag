import sys
import uuid

import click


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (
        not _stream.encoding or _stream.encoding.lower() != "utf-8"
    ):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.ingestion.ingest import ingest as run_ingest
from src.rag.graph import run_multi_agent_query
from src.retrieval.store import count_chunks, delete_all_chunks, existing_sources


console = Console()


@click.group()
def cli() -> None:
    """Greek Legislation RAG — multi-agent CLI."""


@cli.command()
@click.option("--year", type=int, default=None, help="Restrict ingest to one year folder.")
@click.option("--limit", type=int, default=None, help="Process at most N PDFs.")
@click.option("--force", is_flag=True, help="Re-ingest even if source already in DB.")
@click.option("--file", "file", type=str, default=None, help="Ingest a single PDF (filename or path).")
def ingest(year: int | None, limit: int | None, force: bool, file: str | None) -> None:
    """Walk downloads/, parse PDFs with Docling, embed, upsert to Supabase."""
    run_ingest(year=year, limit=limit, force=force, file=file)


@cli.command()
@click.argument("question", required=False)
@click.option("-k", "top_k", type=int, default=None, help="Top K chunks to retrieve.")
@click.option("--year", type=int, default=None, help="Filter to a specific year.")
@click.option("-i", "--interactive", is_flag=True, help="Interactive REPL mode.")
@click.option(
    "--session",
    "session_id",
    type=str,
    default=None,
    help="Session id for multi-turn memory. Defaults to a new UUID per invocation.",
)
def query(
    question: str | None,
    top_k: int | None,
    year: int | None,
    interactive: bool,
    session_id: str | None,
) -> None:
    """Ask the RAG agent a question."""
    sid = session_id or uuid.uuid4().hex

    if interactive:
        console.print(
            f"[bold]Interactive mode.[/bold] Session: [dim]{sid}[/dim]. "
            "Ctrl-C or empty line to exit."
        )
        while True:
            try:
                q = console.input("\n[cyan]>[/cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            _print_answer(
                run_multi_agent_query(q, session_id=sid, year=year, top_k=top_k)
            )
        return

    if not question:
        raise click.UsageError("Provide a question or use --interactive.")
    _print_answer(
        run_multi_agent_query(question, session_id=sid, year=year, top_k=top_k)
    )


def _print_answer(result: dict) -> None:
    console.print(Panel.fit(result["answer"] or "(empty)", title="Answer"))
    rewritten = result.get("rewritten_query")
    if rewritten and rewritten != result.get("query"):
        console.print(f"[dim]Rewritten query: {rewritten}[/dim]")

    chunk_n = len(result.get("chunk_results") or [])
    listing_n = len(result.get("listing_results") or [])
    console.print(
        f"[dim]Agents: rewriter ✓ | chunk ({chunk_n} hits) | "
        f"listing ({listing_n} hits) | combiner ✓[/dim]"
    )

    sources = result.get("sources") or []
    if not sources:
        return
    table = Table(title="Sources", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Source")
    table.add_column("Year")
    table.add_column("Page")
    table.add_column("Sim", justify="right")
    table.add_column("Rank", justify="right")
    for i, s in enumerate(sources, start=1):
        meta = s.metadata or {}
        pages = meta.get("pages") or []
        table.add_row(
            str(i),
            str(meta.get("source", "?")),
            str(meta.get("year", "")),
            str(pages[0]) if pages else "",
            f"{meta.get('similarity', 0):.3f}",
            f"{meta.get('rank', 0):.4f}",
        )
    console.print(table)


@cli.command()
def stats() -> None:
    """Show counts of ingested chunks and sources."""
    chunks = count_chunks()
    sources = existing_sources()
    console.print(f"Chunks: [bold]{chunks}[/bold]")
    console.print(f"Distinct source PDFs: [bold]{len(sources)}[/bold]")
    by_year: dict[str, int] = {}
    for s in sources:
        if len(s) >= 4 and s[:4].isdigit():
            by_year[s[:4]] = by_year.get(s[:4], 0) + 1
    if by_year:
        t = Table(title="By year")
        t.add_column("Year")
        t.add_column("PDFs", justify="right")
        for y in sorted(by_year):
            t.add_row(y, str(by_year[y]))
        console.print(t)


@cli.command()
@click.argument("file", type=str)
@click.option("--no-ocr", is_flag=True, help="Skip the OCR fallback step.")
def diagnose(file: str, no_ocr: bool) -> None:
    """Run text-extraction diagnostics on a PDF (native + OCR)."""
    from src.ingestion.diagnose import diagnose_pdf

    raise SystemExit(diagnose_pdf(file, run_ocr=not no_ocr))


@cli.command()
def reset() -> None:
    """Delete all rows from the documents table (DESTRUCTIVE)."""
    if not click.confirm(
        f"This will delete ALL rows in '{settings.supabase_table}'. Continue?",
        default=False,
    ):
        console.print("Aborted.")
        return
    delete_all_chunks()
    console.print("[green]All chunks deleted.[/green]")


if __name__ == "__main__":
    cli()
