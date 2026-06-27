"""§2.1 Retrieval-recall harness — recall@k / MRR over the synthetic gold set.

Online tier: needs live OpenAI (embeddings) + Supabase (populated corpus). NOT run
on the PR tier (pytest.ini scopes default collection to evals/component). Run with:

    RUN_ONLINE_EVALS=1 pytest evals/retrieval -s

It loads evals/datasets/synthetic_queries.jsonl (produced by synthetic_gen.py),
embeds each question, runs the same match_documents_hybrid path the ChunkAgent
uses, and checks whether the known positive surfaces in top-k.

PRIMARY metric = DOCUMENT-level recall (does the source PDF appear in top-k).
That is what matters for the RAG: the combiner only needs the right document's
chunks, and legislative docs chunk into many near-duplicate siblings, so exact-
chunk recall systematically undercounts (a sibling chunk legitimately outranks
the exact one). Exact-chunk recall@k / MRR / nDCG are kept as SECONDARY ranking
signals. (Diagnostic on the first 30-question set: exact recall@10=0.37 but
same-doc recall@10=0.70 — most exact "misses" were sibling dilution.)
"""

from __future__ import annotations

import math
import os

import pytest

from evals._util import DATASETS_DIR, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
DATASET = DATASETS_DIR / "synthetic_queries.jsonl"

# Regression floors, calibrated to the first clean run (doc recall@10 ≈ 0.70 on a
# 30-question smoke set). Set BELOW observed so the gate guards regression; ratchet
# up as retrieval improves. The genuine ~27% doc-miss rate is a tracked finding
# (see evals/README.md), not something this floor pretends away.
MIN_DOC_RECALL_AT_10 = 0.60

pytestmark = pytest.mark.online


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
@pytest.mark.skipif(not DATASET.exists(), reason="run synthetic_gen.py first")
def test_retrieval_recall(capsys):
    import json

    from src.ingestion.embedder import embed_texts
    from src.retrieval.store import hybrid_search

    from postgrest.exceptions import APIError

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, "empty synthetic gold set"

    K = 10
    exact_hits = {1: 0, 5: 0, 10: 0}
    doc_hits = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0
    # statement_timeout (57014) on the hybrid RPC is itself a tracked signal post-
    # corpus-doubling: a slow query is a recall miss for the user, not a reason to
    # abort the whole tier. Count it as an evaluated doc-miss and record the query.
    timeouts: list[str] = []

    for r in rows:
        target = int(r["chunk_id"])
        target_src = r.get("source")
        emb = embed_texts([r["question"]])
        if not emb:
            continue
        try:
            results = hybrid_search(query_text=r["question"], query_embedding=emb[0], match_count=K)
        except APIError as e:
            if getattr(e, "code", None) == "57014" or "statement timeout" in str(e).lower():
                timeouts.append(r["question"])
                evaluated += 1  # a timeout is a real miss, not a skip
                continue
            raise
        ids = [int(d["id"]) for d in results if d.get("id") is not None]
        srcs = [(d.get("metadata") or {}).get("source") for d in results]
        evaluated += 1

        # PRIMARY: document-level — first rank where the source PDF appears.
        doc_rank = next((i + 1 for i, s in enumerate(srcs) if s == target_src), None)
        if doc_rank:
            for k in doc_hits:
                if doc_rank <= k:
                    doc_hits[k] += 1

        # SECONDARY: exact-chunk ranking quality.
        if target in ids:
            rank = ids.index(target) + 1
            for k in exact_hits:
                if rank <= k:
                    exact_hits[k] += 1
            rr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)  # single relevant -> ideal == 1

    n = evaluated or 1
    report = {
        "n": evaluated,
        "timeouts": len(timeouts),
        "timeout_rate": round(len(timeouts) / n, 4),
        "doc_recall@1": round(doc_hits[1] / n, 4),
        "doc_recall@5": round(doc_hits[5] / n, 4),
        "doc_recall@10": round(doc_hits[10] / n, 4),
        "exact_recall@1": round(exact_hits[1] / n, 4),
        "exact_recall@5": round(exact_hits[5] / n, 4),
        "exact_recall@10": round(exact_hits[10] / n, 4),
        "exact_mrr": round(rr_sum / n, 4),
        "exact_ndcg@10": round(ndcg_sum / n, 4),
    }
    write_report("retrieval_recall.json", report)
    with capsys.disabled():
        print("\n[retrieval]", report)
        if timeouts:
            print(f"[retrieval] {len(timeouts)} statement_timeout(s) on hybrid RPC:")
            for q in timeouts:
                print(f"    - {q}")

    assert report["doc_recall@10"] >= MIN_DOC_RECALL_AT_10, report
