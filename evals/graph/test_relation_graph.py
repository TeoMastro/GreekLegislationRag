"""§2.3 Relational-graph eval (online) — the citation-graph differentiator.

For real focal laws sampled from the graph, fire a natural outgoing relational
question and check that RelationAgent (a) ROUTES to the graph at all
(is_relational + focal-law resolution + direction), and (b) returns the target
laws the graph knows about. Truth comes from the same RPC the agent uses, so a
miss is a routing/hydration failure, not a truth mismatch.

We call RelationAgent.execute directly (1 intent LLM call + graph RPCs + chunk
hydration) rather than the full pipeline — faster, and isolates the relational
path. Run with:
    python -m evals.graph.gen_relation_truth --n 15      # build truth first
    RUN_ONLINE_EVALS=1 pytest evals/graph -s
"""

from __future__ import annotations

import json
import os

import pytest

from evals._util import DATASETS_DIR, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
DATASET = DATASETS_DIR / "relation_graph_truth.jsonl"

MIN_ROUTING_FIRED = 0.85   # natural relational Q must reach the graph
MIN_EDGE_RECALL = 0.85     # of known edges, fraction surfaced (over fired queries)

pytestmark = pytest.mark.online


def _returned_targets(relation_results) -> set[str]:
    out: set[str] = set()
    for d in relation_results:
        rel = (d.metadata or {}).get("_relation_match") or {}
        tgt = rel.get("target_law")
        if tgt:
            out.add(tgt)
    return out


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
@pytest.mark.skipif(not DATASET.exists(), reason="run gen_relation_truth.py first")
def test_relation_graph(capsys):
    from src.rag.agents.relation_agent import get_relation_agent

    agent = get_relation_agent()
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]

    fired = 0
    recalls = []          # per-query edge recall, only where routing fired
    not_fired = []        # focal keys where routing didn't reach the graph
    for r in rows:
        state = {"query": r["query"], "rewritten_query": r["query"], "top_k": None}
        out = agent.execute(state)
        got = _returned_targets(out.get("relation_results") or [])
        if not got:
            not_fired.append(r["focal_key"])
            continue
        fired += 1
        truth = set(r["targets"])
        recalls.append(len(got & truth) / len(truth) if truth else 1.0)

    n = len(rows)
    report = {
        "n": n,
        "routing_fired_rate": round(fired / n, 4) if n else 0.0,
        "edge_recall_mean_over_fired": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "not_fired": not_fired,
    }
    write_report("relation_graph.json", report)
    with capsys.disabled():
        print("\n[relation-graph]", {k: v for k, v in report.items() if k != "not_fired"})
        if not_fired:
            print("[relation-graph] routing did NOT fire for:", not_fired)

    assert report["routing_fired_rate"] >= MIN_ROUTING_FIRED, report
    assert report["edge_recall_mean_over_fired"] is not None and \
        report["edge_recall_mean_over_fired"] >= MIN_EDGE_RECALL, report
