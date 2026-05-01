-- ============================================================
-- Greek Legislation RAG — Supabase init
-- Run in Supabase SQL editor.
-- ============================================================

create extension if not exists vector;

-- ------------------------------------------------------------
-- Chunks table
-- ------------------------------------------------------------
create table if not exists documents (
    id          bigserial primary key,
    content     text not null,
    metadata    jsonb not null default '{}'::jsonb,
    embedding   vector(1536),
    fts         tsvector generated always as (to_tsvector('simple', content)) stored,
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
-- Hybrid match (RRF: semantic + full-text)
-- ------------------------------------------------------------
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
               order by ts_rank_cd(fts, websearch_to_tsquery('simple', query_text)) desc
           ) as rank_ix
    from documents
    where fts @@ websearch_to_tsquery('simple', query_text)
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
