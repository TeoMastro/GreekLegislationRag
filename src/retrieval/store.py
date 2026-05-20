import time
from functools import lru_cache

import httpx
from supabase import Client, create_client

from src.config import settings


@lru_cache(maxsize=1)
def _write_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


@lru_cache(maxsize=1)
def _read_client() -> Client:
    key = settings.supabase_anon_key or settings.supabase_service_key
    return create_client(settings.supabase_url, key)


_TRANSIENT_HTTP_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
)


_INSERT_BATCH_SIZE = 50


def insert_chunks(rows: list[dict]) -> list[int]:
    """Insert chunk rows in batches; return the new ids in insertion order."""
    if not rows:
        return []
    client = _write_client()
    inserted_ids: list[int] = []
    for i in range(0, len(rows), _INSERT_BATCH_SIZE):
        resp = (
            client.table(settings.supabase_table)
            .insert(rows[i : i + _INSERT_BATCH_SIZE])
            .execute()
        )
        for r in resp.data or []:
            rid = r.get("id")
            if rid is not None:
                inserted_ids.append(int(rid))
    return inserted_ids


def delete_by_source(source: str) -> None:
    client = _write_client()
    client.table(settings.supabase_table).delete().eq(
        "metadata->>source", source
    ).execute()


def existing_sources() -> set[str]:
    client = _read_client()
    sources: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table(settings.supabase_table)
            .select("source:metadata->>source")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            src = r.get("source")
            if src:
                sources.add(src)
        if len(rows) < page_size:
            break
        offset += len(rows)
    return sources


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    match_count: int | None = None,
    filter: dict | None = None,
) -> list[dict]:
    params = {
        "query_text": query_text,
        "query_embedding": query_embedding,
        "match_count": match_count or settings.top_k,
        "full_text_weight": settings.hybrid_full_text_weight,
        "semantic_weight": settings.hybrid_semantic_weight,
        "rrf_k": settings.rrf_k,
        "filter": filter or {},
    }
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _read_client().rpc("match_documents_hybrid", params).execute()
            return resp.data or []
        except _TRANSIENT_HTTP_ERRORS:
            _read_client.cache_clear()
            if attempt == max_attempts:
                raise
            time.sleep(0.5 * attempt)
    return []


def count_chunks() -> int:
    client = _read_client()
    resp = (
        client.table(settings.supabase_table)
        .select("id", count="exact")
        .limit(1)
        .execute()
    )
    return resp.count or 0


def delete_all_chunks() -> None:
    client = _write_client()
    client.table(settings.supabase_table).delete().neq("id", -1).execute()


# ---------------------------------------------------------------------------
# Chunk paginators / lookups used by the citation extractor backfill
# ---------------------------------------------------------------------------


def iter_chunks(
    year: int | None = None,
    source: str | None = None,
    page_size: int = 500,
):
    """Yield {id, content, metadata} rows, optionally filtered by year/source.

    Mirrors the existing_sources() pagination pattern (range, page-by-page)
    so it scales to millions of rows without pulling everything at once.
    """
    client = _read_client()
    offset = 0
    while True:
        q = client.table(settings.supabase_table).select("id,content,metadata")
        if year is not None:
            q = q.eq("metadata->>year", str(year))
        if source is not None:
            q = q.eq("metadata->>source", source)
        resp = q.range(offset, offset + page_size - 1).execute()
        rows = resp.data or []
        if not rows:
            return
        for r in rows:
            yield r
        if len(rows) < page_size:
            return
        offset += len(rows)


def fetch_chunks_by_ids(ids: list[int]) -> list[dict]:
    """Fetch chunk rows for a set of ids (used by RelationAgent to hydrate
    citation rows into Document objects). Preserves no particular order."""
    if not ids:
        return []
    client = _read_client()
    out: list[dict] = []
    # Supabase / PostgREST `in` filter is fine for a few hundred ids per call.
    batch = 200
    for i in range(0, len(ids), batch):
        chunk_ids = ids[i : i + batch]
        resp = (
            client.table(settings.supabase_table)
            .select("id,content,metadata")
            .in_("id", chunk_ids)
            .execute()
        )
        out.extend(resp.data or [])
    return out


# ---------------------------------------------------------------------------
# law_nodes
# ---------------------------------------------------------------------------


def upsert_law_node(row: dict) -> int:
    """Upsert on canonical_key; return the law_node id."""
    client = _write_client()
    resp = (
        client.table("law_nodes")
        .upsert(row, on_conflict="canonical_key")
        .execute()
    )
    data = resp.data or []
    if not data:
        # Upsert was a no-op (existing row, nothing changed) — re-select.
        sel = (
            client.table("law_nodes")
            .select("id")
            .eq("canonical_key", row["canonical_key"])
            .limit(1)
            .execute()
        )
        sdata = sel.data or []
        if not sdata:
            raise RuntimeError(
                f"law_nodes upsert returned no row for {row.get('canonical_key')!r}"
            )
        return int(sdata[0]["id"])
    return int(data[0]["id"])


def get_law_node_id_by_source(source: str) -> int | None:
    """Resolve the PDF filename a chunk came from to its parent law_node.id."""
    client = _read_client()
    resp = (
        client.table("law_nodes")
        .select("id")
        .eq("primary_source", source)
        .limit(1)
        .execute()
    )
    data = resp.data or []
    return int(data[0]["id"]) if data else None


def get_law_node_id_by_key(canonical_key: str) -> int | None:
    client = _read_client()
    resp = (
        client.table("law_nodes")
        .select("id")
        .eq("canonical_key", canonical_key)
        .limit(1)
        .execute()
    )
    data = resp.data or []
    return int(data[0]["id"]) if data else None


# ---------------------------------------------------------------------------
# law_citations
# ---------------------------------------------------------------------------


_CITATION_BATCH = 100


def insert_citations(rows: list[dict]) -> int:
    """Insert citation rows, skipping duplicates against the unique constraint.

    The (source_chunk_id, target_law_id, relation, target_article) tuple is
    the natural dedup key. We use upsert with on_conflict on those columns
    and ignore_duplicates=True so re-runs are idempotent.
    Returns the number of rows attempted (not deduped) — caller decides what
    to report.
    """
    if not rows:
        return 0
    client = _write_client()
    total = 0
    for i in range(0, len(rows), _CITATION_BATCH):
        batch = rows[i : i + _CITATION_BATCH]
        # `ignore_duplicates=True` translates to ON CONFLICT DO NOTHING.
        client.table("law_citations").upsert(
            batch,
            on_conflict="source_chunk_id,target_law_id,relation,target_article",
            ignore_duplicates=True,
        ).execute()
        total += len(batch)
    return total


def existing_citation_source_chunk_ids(
    year: int | None = None,
) -> set[int]:
    """Set of chunk ids that already have at least one citation row.

    Used by the backfill CLI to skip chunks already processed in a prior run.
    When `year` is given, narrows to chunks belonging to that year's PDFs by
    joining law_nodes — but PostgREST joins are noisy, so we just return all
    processed chunk_ids and let the caller filter against its own chunk list.
    """
    client = _read_client()
    out: set[int] = set()
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("law_citations")
            .select("source_chunk_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            cid = r.get("source_chunk_id")
            if cid is not None:
                out.add(int(cid))
        if len(rows) < page_size:
            break
        offset += len(rows)
    return out


def mark_chunks_processed(chunk_ids: list[int]) -> None:
    """Record that extract-citations has seen these chunks. Idempotent."""
    if not chunk_ids:
        return
    rows = [{"chunk_id": int(cid)} for cid in chunk_ids]
    client = _write_client()
    batch = 500
    for i in range(0, len(rows), batch):
        client.table("chunk_processing_log").upsert(
            rows[i : i + batch],
            on_conflict="chunk_id",
            ignore_duplicates=True,
        ).execute()


def existing_processed_chunk_ids() -> set[int]:
    """Set of chunk ids extract-citations has already seen."""
    client = _read_client()
    out: set[int] = set()
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("chunk_processing_log")
            .select("chunk_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            cid = r.get("chunk_id")
            if cid is not None:
                out.add(int(cid))
        if len(rows) < page_size:
            break
        offset += len(rows)
    return out


def existing_law_node_keys() -> set[str]:
    """Set of canonical_keys already in law_nodes. Lets sync-law-nodes skip
    listing rows whose node is already present without making any HTTP call."""
    client = _read_client()
    out: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("law_nodes")
            .select("canonical_key")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            k = r.get("canonical_key")
            if k:
                out.add(k)
        if len(rows) < page_size:
            break
        offset += len(rows)
    return out


def citations_from_law(
    canonical_key: str,
    relations: list[str] | None = None,
    min_confidence: float = 0.0,
    limit: int = 50,
) -> list[dict]:
    """RPC wrapper: outgoing edges from `canonical_key`."""
    params = {
        "p_canonical_key": canonical_key,
        "p_relations": relations,
        "p_min_confidence": min_confidence,
        "p_limit": limit,
    }
    resp = _read_client().rpc("citations_from_law", params).execute()
    return resp.data or []


def citations_to_law(
    canonical_key: str,
    relations: list[str] | None = None,
    min_confidence: float = 0.0,
    limit: int = 50,
) -> list[dict]:
    """RPC wrapper: incoming edges to `canonical_key`."""
    params = {
        "p_canonical_key": canonical_key,
        "p_relations": relations,
        "p_min_confidence": min_confidence,
        "p_limit": limit,
    }
    resp = _read_client().rpc("citations_to_law", params).execute()
    return resp.data or []
