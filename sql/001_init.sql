-- ============================================================
-- Greek Legislation RAG — Supabase bootstrap (run once)
-- Run in the Supabase SQL editor on a fresh project.
-- ============================================================

-- ------------------------------------------------------------
-- Extensions
-- ------------------------------------------------------------
create extension if not exists vector;
create extension if not exists unaccent;

-- public.unaccent() is STABLE, which prevents its use inside generated
-- columns or expression indexes. Wrap it as IMMUTABLE so the FTS
-- column and indexes can be built on top of it.
create or replace function f_unaccent(text)
returns text
language sql
immutable
parallel safe
strict
as $$
    select public.unaccent('public.unaccent', $1)
$$;

-- ------------------------------------------------------------
-- Chunks table — content + 1536-d embedding + Greek-stemmed, accent-folded FTS
-- (the 'greek' config Snowball-stems inflections and drops Greek stopwords;
--  f_unaccent keeps it accent-insensitive — see evals/FINDINGS.md #3 Cause B)
-- ------------------------------------------------------------
create table if not exists documents (
    id          bigserial primary key,
    content     text not null,
    metadata    jsonb not null default '{}'::jsonb,
    embedding   vector(1536),
    fts         tsvector generated always as (
                    to_tsvector('greek', f_unaccent(content))
                ) stored,
    created_at  timestamptz not null default now()
);

-- ------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------
create index if not exists documents_embedding_idx
    on documents using hnsw (embedding vector_cosine_ops);

create index if not exists documents_fts_idx
    on documents using gin (fts);

create index if not exists documents_metadata_idx
    on documents using gin (metadata);

create index if not exists documents_source_idx
    on documents ((metadata->>'source'));

-- ------------------------------------------------------------
-- Pure semantic match (cosine)
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
language sql stable
as $$
    select id, content, metadata,
           1 - (embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by embedding <=> query_embedding
    limit match_count;
$$;

-- ------------------------------------------------------------
-- Hybrid match (RRF: semantic + full-text), accent-folded on both sides
-- so inflected Greek forms hit the full-text leg.
--
-- The lexical leg consumes `query_fts`, a pre-built to_tsquery string the
-- caller (src/retrieval/store._build_fts_query) ORs together from the query's
-- distinctive, stopword-filtered terms. websearch_to_tsquery ANDed every term,
-- so a natural-language question became a conjunction no chunk satisfied (0 FTS
-- rows → silently pure-semantic; see evals/FINDINGS.md #3). The 'greek' config
-- Snowball-stems both sides, so inflected query terms hit stored inflections
-- (the prior 'simple' config did no stemming — #3 Cause B). `query_text` is kept
-- for API compatibility / debugging. Empty `query_fts` ⇒ pure-semantic.
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

-- ------------------------------------------------------------
-- Role config
-- Required on Supabase free tier: the default 8s per-statement
-- timeout cancels bulk inserts of OCR'd PDFs (180+ chunks of
-- vector(1536)) while the HNSW index updates.
-- ------------------------------------------------------------
alter role service_role set statement_timeout = '60s';
