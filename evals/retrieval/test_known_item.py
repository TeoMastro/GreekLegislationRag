"""§2.2 Known-item retrieval eval (online) — the ListingAgent's structured path.

A query that names a specific document ("Ν. 5263") must return chunks from exactly
that ΦΕΚ. This exercises ListingAgent.execute end-to-end: intent extraction →
candidate selection over the manifests → scoped per-PDF hybrid search. Distinct
from §2.1 (semantic recall over the whole corpus) and §4 (intent parsing only).

Online tier — needs OpenAI + Supabase + local listing manifests. Run with:
    python -m evals.retrieval.gen_known_item --n 15     # build truth first
    RUN_ONLINE_EVALS=1 pytest evals/retrieval/test_known_item.py -s
"""

from __future__ import annotations

import json
import os

import pytest

from evals._util import DATASETS_DIR, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
DATASET = DATASETS_DIR / "known_item.jsonl"

MIN_HIT_RATE = 0.85    # expected PDF appears among listing results
MIN_TOP1_RATE = 0.70   # expected PDF is the top listing result

pytestmark = pytest.mark.online


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
@pytest.mark.skipif(not DATASET.exists(), reason="run gen_known_item.py first")
def test_known_item(capsys):
    from src.ingestion.embedder import embed_texts
    from src.rag.agents.listing_agent import get_listing_agent

    agent = get_listing_agent()
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]

    hits = top1 = evaluated = 0
    misses = []
    for r in rows:
        emb = embed_texts([r["query"]])
        if not emb:
            continue
        evaluated += 1
        state = {"query": r["query"], "rewritten_query": r["query"],
                 "query_embedding": emb[0], "year": None, "top_k": None}
        out = agent.execute(state)
        results = out.get("listing_results") or []
        sources = [(d.metadata or {}).get("source") for d in results]
        expected = r["expected_source"]
        if expected in sources:
            hits += 1
        else:
            misses.append({"query": r["query"], "expected": expected, "got": sources[:3]})
        if sources and sources[0] == expected:
            top1 += 1

    n = evaluated or 1
    report = {
        "n": evaluated,
        "hit_rate": round(hits / n, 4),
        "top1_rate": round(top1 / n, 4),
        "misses": misses,
    }
    write_report("known_item.json", report)
    with capsys.disabled():
        print("\n[known-item]", {k: v for k, v in report.items() if k != "misses"})
        if misses:
            print("[known-item] misses:", misses)

    assert report["hit_rate"] >= MIN_HIT_RATE, report
    assert report["top1_rate"] >= MIN_TOP1_RATE, report
