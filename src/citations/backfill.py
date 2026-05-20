"""CLI-facing orchestration for the citation graph.

Two operations:
  * sync_law_nodes — walks listing-items-*.md (via ListingAgent's existing
    parser) and upserts one law_node per row.
  * extract_citations — paginates `documents`, runs the regex+LLM extractor,
    upserts target law_nodes, inserts law_citations. Resumable.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.citations.extractor import find_candidates, classify_citations
from src.citations.normalize import canonical_key, canonicalize_kind
from src.rag.agents.listing_agent import _load_listings  # type: ignore[attr-defined]
from src.retrieval.store import (
    existing_citation_source_chunk_ids,
    existing_law_node_keys,
    existing_processed_chunk_ids,
    get_law_node_id_by_key,
    get_law_node_id_by_source,
    insert_citations,
    iter_chunks,
    mark_chunks_processed,
    upsert_law_node,
)


console = Console()


def sync_law_nodes(force: bool = False) -> int:
    """Upsert law_nodes from every listing-items-YYYY.md row.

    Skips rows whose canonical_key is already in law_nodes (one batched
    SELECT instead of one upsert per row). Pass force=True to re-upsert
    everything — needed if the listing files were edited.
    """
    rows = _load_listings()
    if not rows:
        console.print("[yellow]No listing rows found under downloads/.[/yellow]")
        return 0

    already = set() if force else existing_law_node_keys()
    upserted = 0
    already_present = 0
    skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing law_nodes", total=len(rows))
        for r in rows:
            kind = canonicalize_kind(r.get("kind") or "")
            number = r.get("number")
            year = r.get("fek_year")
            if year is None and r.get("fek_date"):
                year = r["fek_date"].year
            if kind is None or number is None or year is None:
                skipped += 1
                progress.advance(task)
                continue
            key = canonical_key(kind, int(number), int(year))
            if key in already:
                already_present += 1
                progress.advance(task)
                continue
            node = {
                "canonical_key": key,
                "kind": kind,
                "number": int(number),
                "year": int(year),
                "fek_series": r.get("fek_series"),
                "fek_number": r.get("fek_number"),
                "fek_year": r.get("fek_year"),
                "title": r.get("description") or None,
                "primary_source": r.get("pdf_basename") or None,
            }
            try:
                upsert_law_node(node)
                upserted += 1
                already.add(key)
            except Exception as e:
                console.print(f"[yellow]upsert failed for {key}: {e}[/yellow]")
                skipped += 1
            progress.advance(task)

    console.print(
        f"[green]Done.[/green] law_nodes upserted: {upserted}, "
        f"already present (skipped): {already_present}, "
        f"unparseable (skipped): {skipped}"
    )
    return upserted


def _resolve_source_law(
    source: str,
    cache: dict[str, int | None],
) -> int | None:
    if source in cache:
        return cache[source]
    nid = get_law_node_id_by_source(source)
    cache[source] = nid
    return nid


def _resolve_target_law(
    kind: str,
    number: int,
    year: int,
    cache: dict[str, int],
) -> int:
    key = canonical_key(kind, number, year)
    if key in cache:
        return cache[key]
    nid = get_law_node_id_by_key(key)
    if nid is None:
        # Cited-but-not-ingested law: create a stub so the edge can be stored.
        nid = upsert_law_node(
            {
                "canonical_key": key,
                "kind": kind,
                "number": number,
                "year": year,
                # primary_source intentionally NULL — will be filled in if the
                # law's own PDF gets ingested later and sync-law-nodes is rerun.
            }
        )
    cache[key] = nid
    return nid


def extract_for_chunks(
    source_pdf: str,
    chunks: list[tuple[int, str]],
) -> int:
    """Extract citations from a freshly-ingested set of (chunk_id, text) pairs.

    Called by the ingest pipeline immediately after insert_chunks, so the
    source_law_id MUST already exist in law_nodes (sync-law-nodes must have
    been run, or the listing item must be in scope of the current sync).
    Returns the number of citation rows inserted (before dedup).
    """
    if not chunks:
        return 0

    source_law_id = get_law_node_id_by_source(source_pdf)
    if source_law_id is None:
        console.print(
            f"[yellow]No law_node for {source_pdf} — skipping citation extraction. "
            f"Run `sync-law-nodes` first.[/yellow]"
        )
        return 0

    target_law_cache: dict[str, int] = {}
    rows: list[dict] = []
    seen_chunk_ids: list[int] = []
    for chunk_id, text in chunks:
        seen_chunk_ids.append(chunk_id)
        cands = find_candidates(text)
        if not cands:
            continue
        for c in classify_citations(text, cands):
            target_id = _resolve_target_law(
                c["target_kind"],
                c["target_number"],
                c["target_year"],
                target_law_cache,
            )
            rows.append(
                {
                    "source_law_id": source_law_id,
                    "source_chunk_id": chunk_id,
                    "target_law_id": target_id,
                    "target_article": c.get("target_article") or "",
                    "relation": c["relation"],
                    "snippet": c["snippet"],
                    "confidence": c["confidence"],
                }
            )
    if rows:
        insert_citations(rows)
    # Mark every chunk we examined (even ones with no candidates) so a future
    # `extract-citations` run does not rescan them.
    mark_chunks_processed(seen_chunk_ids)
    return len(rows)


def extract_citations(
    year: int | None = None,
    source: str | None = None,
    skip_already_processed: bool = True,
) -> dict[str, int]:
    """Walk documents (optionally year/source filtered), extract citations.

    Returns a dict with counters: scanned, candidates, edges_written, skipped.
    """
    if skip_already_processed:
        # Skip a chunk if (a) we marked it processed previously OR (b) it has
        # citation rows from before the marker table existed. The union keeps
        # legacy runs from re-doing work.
        already = existing_processed_chunk_ids() | existing_citation_source_chunk_ids()
    else:
        already = set()

    source_law_cache: dict[str, int | None] = {}
    target_law_cache: dict[str, int] = {}

    scanned = 0
    with_candidates = 0
    edges_written = 0
    missing_source_law = 0

    pending: list[dict] = []
    pending_marks: list[int] = []
    BATCH = 25         # citation rows; low so progress is visible quickly.
    MARK_BATCH = 200   # chunk markers; bigger because most chunks have no
                       # citations and we want fewer round-trips.

    def flush_citations() -> None:
        nonlocal pending, edges_written
        if not pending:
            return
        try:
            insert_citations(pending)
            edges_written += len(pending)
        except Exception as e:
            console.print(f"[red]citation insert batch failed:[/red] {e}")
        pending = []

    def flush_marks() -> None:
        nonlocal pending_marks
        if not pending_marks:
            return
        try:
            mark_chunks_processed(pending_marks)
        except Exception as e:
            console.print(f"[red]chunk-marker insert batch failed:[/red] {e}")
        pending_marks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[written]} edges"),
        TextColumn("{task.completed} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting citations", total=None, written=0)

        for row in iter_chunks(year=year, source=source):
            scanned += 1
            progress.advance(task)
            chunk_id = int(row["id"])
            if chunk_id in already:
                continue

            # From here on this chunk is being scanned for real — mark it
            # processed regardless of whether anything comes out, so a future
            # run does not rescan it.
            pending_marks.append(chunk_id)

            text = row.get("content") or ""
            cands = find_candidates(text)
            if cands:
                with_candidates += 1
                meta = row.get("metadata") or {}
                src_pdf = meta.get("source")
                if src_pdf:
                    source_law_id = _resolve_source_law(src_pdf, source_law_cache)
                    if source_law_id is None:
                        missing_source_law += 1
                    else:
                        citations = classify_citations(text, cands)
                        for c in citations:
                            target_id = _resolve_target_law(
                                c["target_kind"],
                                c["target_number"],
                                c["target_year"],
                                target_law_cache,
                            )
                            pending.append(
                                {
                                    "source_law_id": source_law_id,
                                    "source_chunk_id": chunk_id,
                                    "target_law_id": target_id,
                                    "target_article": c.get("target_article") or "",
                                    "relation": c["relation"],
                                    "snippet": c["snippet"],
                                    "confidence": c["confidence"],
                                }
                            )
                            if len(pending) >= BATCH:
                                flush_citations()
                                progress.update(task, written=edges_written)

            if len(pending_marks) >= MARK_BATCH:
                flush_marks()

        flush_citations()
        flush_marks()
        progress.update(task, written=edges_written)

    if missing_source_law:
        console.print(
            f"[yellow]{missing_source_law} chunks had no matching law_node "
            f"for their source PDF — did you run sync-law-nodes first?[/yellow]"
        )
    console.print(
        f"[green]Done.[/green] scanned={scanned}, with_candidates={with_candidates}, "
        f"edges_written={edges_written}"
    )
    return {
        "scanned": scanned,
        "with_candidates": with_candidates,
        "edges_written": edges_written,
        "missing_source_law": missing_source_law,
    }
