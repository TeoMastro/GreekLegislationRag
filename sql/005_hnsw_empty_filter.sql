-- ============================================================
-- Greek Legislation RAG — restore HNSW on the unfiltered semantic path
-- (run once, after 001–004). See evals/FINDINGS.md.
--
-- Symptom: after the corpus doubled to ~105k chunks, EVERY full-corpus query
-- (pure semantic AND hybrid) hit `statement_timeout` (57014) at ~8s. The
-- retrieval eval went to doc_recall@10 = 0.0 / timeout_rate = 1.0.
--
-- Cause: match_documents / the hybrid semantic CTE both carry
--   WHERE metadata @> filter
-- and the caller passes filter = '{}' for the common unscoped query.
-- `metadata @> '{}'` matches every row and is GIN-indexable, so the planner
-- builds the candidate set from documents_metadata_idx (all 105k rows) and then
-- does an EXACT distance sort — the HNSW index (documents_embedding_idx) is
-- never consulted. This was always brute-force KNN; it merely fit under the 8s
-- timeout at ~57k (old scale-probe p50 4.8s) and crossed it once the corpus
-- doubled. A bare `ORDER BY embedding <=> q LIMIT k` (no WHERE) uses HNSW and
-- returns in ~70ms — proven via EXPLAIN ANALYZE.
--
-- Fix: branch on an empty filter. When filter = '{}', run the clean
-- ORDER BY … LIMIT the HNSW index serves. When a real filter is supplied
-- (e.g. {"source": "…"} from the ListingAgent's scoped path), keep the
-- containment predicate — that candidate set is tiny, so a filtered exact sort
-- is fast and correct. Implemented in plpgsql so the WHERE can be conditional.
--
-- Functions only — no data is touched. statement_timeout is also re-asserted
-- on service_role (001 set it but it is not in effect on this instance); the
-- HNSW path makes 8s moot, but the larger ceiling protects scoped/edge queries.
-- ============================================================

alter role service_role set statement_timeout = '60s';

-- ------------------------------------------------------------
-- Pure semantic match — HNSW on empty filter, exact on scoped filter
-- ------------------------------------------------------------
create or replace function match_documents (
    query_embedding vector(1536),
    match_count     int   default 10,
    filter          jsonb default '{}'::jsonb
)
returns table (
    id          bigint,
    content     text,
    metadata    jsonb,
    similarity  float
)
language plpgsql stable
as $$
begin
    if filter = '{}'::jsonb then
        return query
            select d.id, d.content, d.metadata,
                   1 - (d.embedding <=> query_embedding) as similarity
            from documents d
            order by d.embedding <=> query_embedding
            limit match_count;
    else
        return query
            select d.id, d.content, d.metadata,
                   1 - (d.embedding <=> query_embedding) as similarity
            from documents d
            where d.metadata @> filter
            order by d.embedding <=> query_embedding
            limit match_count;
    end if;
end;
$$;

-- ------------------------------------------------------------
-- Hybrid match (RRF: semantic + full-text). The semantic leg must run as a
-- literally separate WHERE-less query on the empty-filter path, or the HNSW
-- index is skipped — a `WHERE filter='{}' OR metadata @> filter` predicate
-- does NOT work, because inside a plpgsql cached plan `filter` is a runtime
-- parameter the planner can't fold to a constant, so it stays a runtime filter
-- and brute-forces (the exact bug this migration fixes). Hence explicit
-- IF/ELSE branches. The full_text CTE is bounded by `fts @@ to_tsquery(...)`
-- (GIN-selective) so it keeps the containment predicate either way. Lexical
-- config stays 'greek' (004). RRF fusion math is identical in both branches.
-- ------------------------------------------------------------
create or replace function match_documents_hybrid (
    query_text          text,
    query_embedding     vector(1536),
    match_count         int   default 10,
    full_text_weight    float default 1.0,
    semantic_weight     float default 1.0,
    rrf_k               int   default 50,
    filter              jsonb default '{}'::jsonb,
    query_fts           text  default ''
)
returns table (
    id          bigint,
    content     text,
    metadata    jsonb,
    similarity  float,
    rank        float
)
language plpgsql stable
as $$
begin
    if filter = '{}'::jsonb then
        return query
        with full_text as (
            select d.id,
                   row_number() over (
                       order by ts_rank_cd(d.fts, to_tsquery('greek', f_unaccent(query_fts))) desc
                   ) as rank_ix
            from documents d
            where query_fts <> ''
              and d.fts @@ to_tsquery('greek', f_unaccent(query_fts))
            limit least(match_count * 2, 50)
        ),
        semantic as (
            -- No WHERE ⇒ HNSW index scan on documents_embedding_idx.
            select d.id,
                   row_number() over (order by d.embedding <=> query_embedding) as rank_ix
            from documents d
            order by d.embedding <=> query_embedding
            limit least(match_count * 2, 50)
        )
        select  d.id, d.content, d.metadata,
                1 - (d.embedding <=> query_embedding) as similarity,
                coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight
              + coalesce(1.0 / (rrf_k +  s.rank_ix), 0.0) * semantic_weight as rank
        from full_text ft
        full outer join semantic s using (id)
        join documents d on d.id = coalesce(ft.id, s.id)
        order by rank desc
        limit match_count;
    else
        return query
        with full_text as (
            select d.id,
                   row_number() over (
                       order by ts_rank_cd(d.fts, to_tsquery('greek', f_unaccent(query_fts))) desc
                   ) as rank_ix
            from documents d
            where query_fts <> ''
              and d.fts @@ to_tsquery('greek', f_unaccent(query_fts))
              and d.metadata @> filter
            limit least(match_count * 2, 50)
        ),
        semantic as (
            -- Scoped: small candidate set ⇒ fast exact sort.
            select d.id,
                   row_number() over (order by d.embedding <=> query_embedding) as rank_ix
            from documents d
            where d.metadata @> filter
            order by d.embedding <=> query_embedding
            limit least(match_count * 2, 50)
        )
        select  d.id, d.content, d.metadata,
                1 - (d.embedding <=> query_embedding) as similarity,
                coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight
              + coalesce(1.0 / (rrf_k +  s.rank_ix), 0.0) * semantic_weight as rank
        from full_text ft
        full outer join semantic s using (id)
        join documents d on d.id = coalesce(ft.id, s.id)
        order by rank desc
        limit match_count;
    end if;
end;
$$;
