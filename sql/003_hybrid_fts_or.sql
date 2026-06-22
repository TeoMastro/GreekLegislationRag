-- ============================================================
-- Greek Legislation RAG — fix the hybrid lexical leg (run once, after 001).
-- See evals/FINDINGS.md #3.
--
-- Before: match_documents_hybrid built its full-text leg with
--   websearch_to_tsquery('simple', query_text)
-- which ANDs every query term. A natural-language question became a
-- many-term conjunction no chunk satisfied → 0 FTS rows → the hybrid result
-- was byte-identical to pure cosine, and full_text_weight/rrf_k were inert.
--
-- After: the leg consumes a pre-built `query_fts` to_tsquery string that the
-- caller (src/retrieval/store._build_fts_query) ORs from the query's
-- distinctive, stopword-filtered terms (alpha terms get a `:*` prefix for
-- partial Greek-inflection matching; ubiquitous terms like άρθρο/νόμος are
-- dropped to keep the candidate scan bounded — they were the statement_timeout
-- culprits). Empty `query_fts` ⇒ pure-semantic, matching prior de-facto behaviour.
--
-- Adding `query_fts` changes the argument signature, so CREATE OR REPLACE would
-- leave the old 7-arg function as a lingering overload — drop it first. No data
-- is touched (functions only).
-- ============================================================

drop function if exists match_documents_hybrid(
    text, vector, int, float, float, int, jsonb
);

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
                   to_tsquery('simple', f_unaccent(query_fts))
               ) desc
           ) as rank_ix
    from documents
    where query_fts <> ''
      and fts @@ to_tsquery('simple', f_unaccent(query_fts))
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
