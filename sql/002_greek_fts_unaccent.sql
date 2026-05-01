-- ============================================================
-- Greek Legislation RAG — Greek FTS recall fix
-- Folds accents on both indexed text and query so that
-- inflected Greek forms hit the full-text leg of hybrid search.
-- Run in Supabase SQL editor AFTER 001_init.sql.
-- ============================================================

create extension if not exists unaccent;

-- public.unaccent() is STABLE, which prevents using it inside generated
-- columns or expression indexes. Wrap it as IMMUTABLE.
create or replace function f_unaccent(text)
returns text
language sql
immutable
parallel safe
strict
as $$
    select public.unaccent('public.unaccent', $1)
$$;

-- Generated stored columns can't be ALTERed; drop and re-add.
-- The GIN index drops with the column.
drop index if exists documents_fts_idx;
alter table documents drop column if exists fts;
alter table documents add column fts tsvector
    generated always as (to_tsvector('simple', f_unaccent(content))) stored;
create index documents_fts_idx on documents using gin (fts);

-- Apply f_unaccent on the query side as well.
create or replace function match_documents_hybrid (
    query_text          text,
    query_embedding     vector(1536),
    match_count         int   default 10,
    full_text_weight    float default 1.0,
    semantic_weight     float default 1.0,
    rrf_k               int   default 50,
    filter              jsonb default '{}'::jsonb
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
                   websearch_to_tsquery('simple', f_unaccent(query_text))
               ) desc
           ) as rank_ix
    from documents
    where fts @@ websearch_to_tsquery('simple', f_unaccent(query_text))
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
