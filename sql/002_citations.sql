-- ============================================================
-- Greek Legislation RAG — citation graph (run once, after 001).
-- Adds law_nodes + law_citations on top of the documents table.
-- ============================================================

-- ------------------------------------------------------------
-- Nodes: one canonical entry per law, regardless of inflection.
-- canonical_key collapses 'ν./Ν./νόμος/νόμου' etc. into one form.
-- primary_source is the PDF filename when the law's own publication
-- is in the corpus, NULL for cited-but-not-ingested laws.
-- ------------------------------------------------------------
create table if not exists law_nodes (
    id              bigserial primary key,
    canonical_key   text unique not null,
    kind            text not null,
    number          int  not null,
    year            int  not null,
    fek_series      text,
    fek_number      int,
    fek_year        int,
    title           text,
    primary_source  text,
    created_at      timestamptz not null default now()
);

create index if not exists law_nodes_kind_number_year_idx
    on law_nodes (kind, number, year);
create index if not exists law_nodes_primary_source_idx
    on law_nodes (primary_source);

-- ------------------------------------------------------------
-- Edges: extracted at ingest time from chunk text. Each row points
-- back to the chunk it was extracted from so we can fetch context.
-- ------------------------------------------------------------
create table if not exists law_citations (
    id                  bigserial primary key,
    source_law_id       bigint not null references law_nodes(id) on delete cascade,
    source_chunk_id     bigint not null references documents(id) on delete cascade,
    target_law_id       bigint not null references law_nodes(id) on delete cascade,
    target_article      text   not null default '',  -- '' = no article specified
    relation            text   not null,
    snippet             text   not null,
    confidence          float  not null,
    created_at          timestamptz not null default now()
);

-- ------------------------------------------------------------
-- Self-healing migration (idempotent).
--
-- An earlier version of this file used a unique INDEX with a
-- coalesce(target_article, '') expression. PostgREST's upsert with
-- on_conflict cannot match an expression index, so every insert from
-- the Python client failed with SQLSTATE 42P10. The block below
-- repairs that, AND is a no-op on fresh installs.
-- ------------------------------------------------------------

-- 1. Drop the bad expression index from the v1 of this file, if present.
drop index if exists law_citations_dedup_idx;

-- 2. If the v1 table had target_article as nullable, backfill NULLs and
--    tighten the column shape. These ALTER statements are no-ops when the
--    column already matches.
update law_citations set target_article = '' where target_article is null;
alter table law_citations alter column target_article set default '';
alter table law_citations alter column target_article set not null;

-- 3. Ensure a plain-column unique constraint exists (this is what PostgREST
--    needs to make ignore_duplicates=True upserts work). Skip if a unique
--    constraint already covers the same four columns under any name —
--    avoids creating a duplicate when CREATE TABLE on an even older revision
--    left an auto-named constraint behind.
do $$
declare
    target_cols text[] := array['relation', 'source_chunk_id', 'target_article', 'target_law_id'];
    existing_cols text[];
    c record;
    have_match boolean := false;
begin
    for c in
        select conname, conkey
        from pg_constraint
        where conrelid = 'law_citations'::regclass
          and contype = 'u'
    loop
        select array_agg(attname order by attname)
          into existing_cols
        from pg_attribute
        where attrelid = 'law_citations'::regclass
          and attnum = any(c.conkey);
        if existing_cols = target_cols then
            have_match := true;
            exit;
        end if;
    end loop;

    if not have_match then
        alter table law_citations
            add constraint law_citations_uniq
            unique (source_chunk_id, target_law_id, relation, target_article);
    end if;
end$$;

create index if not exists law_citations_source_idx       on law_citations (source_law_id);
create index if not exists law_citations_target_idx       on law_citations (target_law_id);
create index if not exists law_citations_relation_idx     on law_citations (relation);
create index if not exists law_citations_source_chunk_idx on law_citations (source_chunk_id);

-- ------------------------------------------------------------
-- chunk_processing_log: marker per (chunk_id) that extract-citations
-- has seen. Used so re-runs skip chunks regardless of whether they
-- produced any law_citations rows (most don't — regex pre-filter
-- rejects them — but we still want to avoid rescanning them).
-- ------------------------------------------------------------
create table if not exists chunk_processing_log (
    chunk_id               bigint primary key references documents(id) on delete cascade,
    citations_extracted_at timestamptz not null default now()
);

-- ------------------------------------------------------------
-- RPC: outgoing edges from a law ("what does X amend / repeal / cite")
-- ------------------------------------------------------------
create or replace function citations_from_law (
    p_canonical_key text,
    p_relations     text[] default null,
    p_min_confidence float default 0.0,
    p_limit         int    default 50
)
returns table (
    citation_id     bigint,
    source_chunk_id bigint,
    target_key      text,
    target_kind     text,
    target_number   int,
    target_year     int,
    target_article  text,
    relation        text,
    snippet         text,
    confidence      float
)
language sql stable
as $$
    select c.id, c.source_chunk_id,
           t.canonical_key, t.kind, t.number, t.year,
           nullif(c.target_article, '') as target_article,
           c.relation, c.snippet, c.confidence
    from law_citations c
    join law_nodes s on s.id = c.source_law_id
    join law_nodes t on t.id = c.target_law_id
    where s.canonical_key = p_canonical_key
      and c.confidence >= p_min_confidence
      and (p_relations is null or c.relation = any(p_relations))
    order by c.confidence desc, c.id
    limit p_limit;
$$;

-- ------------------------------------------------------------
-- RPC: incoming edges to a law ("which laws affected X")
-- ------------------------------------------------------------
create or replace function citations_to_law (
    p_canonical_key text,
    p_relations     text[] default null,
    p_min_confidence float default 0.0,
    p_limit         int    default 50
)
returns table (
    citation_id     bigint,
    source_chunk_id bigint,
    source_key      text,
    source_kind     text,
    source_number   int,
    source_year     int,
    target_article  text,
    relation        text,
    snippet         text,
    confidence      float
)
language sql stable
as $$
    select c.id, c.source_chunk_id,
           s.canonical_key, s.kind, s.number, s.year,
           nullif(c.target_article, '') as target_article,
           c.relation, c.snippet, c.confidence
    from law_citations c
    join law_nodes s on s.id = c.source_law_id
    join law_nodes t on t.id = c.target_law_id
    where t.canonical_key = p_canonical_key
      and c.confidence >= p_min_confidence
      and (p_relations is null or c.relation = any(p_relations))
    order by c.confidence desc, c.id
    limit p_limit;
$$;
