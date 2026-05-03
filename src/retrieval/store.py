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


def insert_chunks(rows: list[dict]) -> None:
    if not rows:
        return
    client = _write_client()
    for i in range(0, len(rows), _INSERT_BATCH_SIZE):
        client.table(settings.supabase_table).insert(
            rows[i : i + _INSERT_BATCH_SIZE]
        ).execute()


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
