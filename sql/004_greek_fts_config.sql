-- ============================================================
-- Greek Legislation RAG — switch the FTS leg from the 'simple' text-search
-- config to 'greek' (run once, after 001/002/003). See evals/FINDINGS.md #3.
--
-- Why: 'simple' does NO stemming, so an inflected query term ("νόμος") never
-- matched a stored inflection ("νόμου") — the open half of #3 (Cause B). The
-- old `:*` prefix in _build_fts_query only helped when the stored form EXTENDED
-- the query form. The 'greek' Snowball config stems both sides, so inflections
-- match in every direction. Verified on live data: of 100 chunks containing the
-- inflected «νόμου», a «νόμος» query matched 97 under 'greek' vs 0 under 'simple'.
-- f_unaccent is kept on both sides — it does not break stemming (98/100) and
-- preserves accent-insensitivity (the caller folds the query side identically).
--
-- A stored GENERATED column's expression can't be altered in place on PG 15, so
-- we drop and re-add `fts`. Re-adding regenerates it from `content` for every
-- existing row (no data loss — `content` is the source) and we rebuild the GIN
-- index. This takes a brief ACCESS EXCLUSIVE lock + a one-time rewrite/reindex
-- of the documents table (seconds-to-minutes at ~57k rows). `embedding` and the
-- HNSW index are untouched.
-- ============================================================

-- 1. Drop the dependent GIN index and the generated column, then re-add the
--    column with the 'greek' config and recreate the index.
drop index if exists documents_fts_idx;

alter table documents drop column if exists fts;

alter table documents
    add column fts tsvector generated always as (
        to_tsvector('greek', f_unaccent(content))
    ) stored;

create index documents_fts_idx on documents using gin (fts);

-- 2. Repoint the hybrid RPC's lexical leg at the 'greek' config. Signature is
--    unchanged (8 args), so CREATE OR REPLACE is enough — no drop needed.
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
language sql stable
as $$
with full_text as (
    select id,
           row_number() over (
               order by ts_rank_cd(
                   fts,
                   to_tsquery('greek', f_unaccent(query_fts))
               ) desc
           ) as rank_ix
    from documents
    where query_fts <> ''
      and fts @@ to_tsquery('greek', f_unaccent(query_fts))
      and metadata @> filter
    limit least(match_count * 2, 50)
),
semantic as (
    select id,
           row_number() over (order by embedding <=> query_embedding) as rank_ix
    from documents
    where metadata @> filter
    limit least(match_count * 2, 50)
)
select  d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity,
        coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight
      + coalesce(1.0 / (rrf_k +  s.rank_ix), 0.0) * semantic_weight as rank
from full_text ft
full outer join semantic s using (id)
join documents d on d.id = coalesce(ft.id, s.id)
order by rank desc
limit match_count;
$$;
