from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import settings
from src.ingestion.ingest import _extract, assess_text_quality


_console = Console()


def _greek_count(text: str) -> int:
    return sum(
        1 for c in text
        if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿"
    )


def _latin_count(text: str) -> int:
    return sum(1 for c in text if "a" <= c.lower() <= "z")


def _resolve_pdf(arg: str) -> Path | None:
    p = Path(arg)
    if p.is_file():
        return p
    matches = [
        m for m in settings.downloads_dir.glob(f"**/{arg}")
        if ".ocr" not in m.parts
    ]
    if not matches and not arg.lower().endswith(".pdf"):
        matches = [
            m for m in settings.downloads_dir.glob(f"**/{arg}.pdf")
            if ".ocr" not in m.parts
        ]
    return matches[0] if matches else None


def _report(label: str, pdf: Path) -> bool:
    chunks, total_pages = _extract(pdf)
    total_chars = sum(len(c["text"].strip()) for c in chunks)
    pages_with_text = len({p for c in chunks for p in (c.get("pages") or [])})
    full_text = "\n".join(c["text"] for c in chunks)
    text_len = max(len(full_text), 1)
    greek = _greek_count(full_text)
    latin = _latin_count(full_text)
    ok, reason = assess_text_quality(chunks, total_pages)

    table = Table(title=f"{label}: {pdf.name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row(
        "status",
        "[green]PASS[/green]" if ok else f"[red]FAIL[/red] — {reason}",
    )
    table.add_row("total chars", f"{total_chars:,}")
    table.add_row("total pages", str(total_pages))
    table.add_row("pages with text", f"{pages_with_text}/{total_pages}")
    if total_pages:
        table.add_row("chars/page", f"{total_chars / total_pages:.0f}")
    table.add_row(
        "greek chars",
        f"{greek:,} ({greek * 100 / text_len:.1f}%)",
    )
    table.add_row(
        "latin chars",
        f"{latin:,} ({latin * 100 / text_len:.1f}%)",
    )
    table.add_row("chunks", str(len(chunks)))
    _console.print(table)

    sample = full_text[:600].replace("\n", " ")
    if sample:
        _console.print(f"\n[dim]Sample (first 600 chars):[/dim]\n{sample}\n")
    return ok


def diagnose_pdf(arg: str, run_ocr: bool = True) -> int:
    pdf = _resolve_pdf(arg)
    if pdf is None:
        _console.print(f"[red]File not found: {arg}[/red]")
        return 1

    _console.print(f"[bold]Diagnosing[/bold] {pdf}\n")
    ok = _report("Native (text layer)", pdf)

    if not run_ocr:
        return 0
    if ok:
        _console.print("[green]Native extraction passed; skipping OCR.[/green]")
        return 0

    _console.print("\n[bold]Running OCR via ocrmypdf...[/bold]\n")
    try:
        from src.pdf.ocr import ensure_ocr

        ocr_pdf = ensure_ocr(pdf)
    except Exception as e:
        _console.print(f"[red]OCR failed: {e}[/red]")
        return 1

    _report("After OCR", ocr_pdf)
    return 0
