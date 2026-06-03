"""§4.1 RelationAgent intent eval (online) — does routing fire correctly?

A false negative on `is_relational` is the costly error: the question falls back
to lexical retrieval that (per the README) structurally cannot answer a relation
question. So we report is_relational recall separately, plus direction accuracy
and resolved-focal-key accuracy over the truly-relational queries.

Online tier — needs OpenAI. NOT collected on the PR tier. Run with:
    RUN_ONLINE_EVALS=1 pytest evals/routing/test_relation_intent.py -s
"""

from __future__ import annotations

import os

import pytest

from evals._util import load_jsonl, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
GOLD = load_jsonl("relation_intent.jsonl")

# Regression floors — calibrate to the first clean run, then ratchet up.
MIN_IS_RELATIONAL_ACC = 0.85
MIN_IS_RELATIONAL_RECALL = 0.90   # missing a relational query is the expensive error
MIN_DIRECTION_ACC = 0.75
MIN_FOCAL_KEY_ACC = 0.90

pytestmark = pytest.mark.online


def _expected_focal_key(ra, row):
    sl = row.get("source_law")
    return ra._resolve_focal_law({"source_law": sl}) if sl else None


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
def test_relation_intent(capsys):
    from src.rag.agents.relation_agent import get_relation_agent

    ra = get_relation_agent()

    is_rel_correct = 0
    rel_tp = rel_fn = 0          # for is_relational recall
    dir_correct = dir_total = 0
    key_correct = key_total = 0
    failures = []

    for row in GOLD:
        intent = ra._extract_intent(row["query"])
        pred_is_rel = bool(intent.get("is_relational"))
        exp_is_rel = bool(row["is_relational"])

        if pred_is_rel == exp_is_rel:
            is_rel_correct += 1
        else:
            failures.append({"query": row["query"], "exp_is_rel": exp_is_rel, "got": pred_is_rel})

        if exp_is_rel:
            if pred_is_rel:
                rel_tp += 1
            else:
                rel_fn += 1

            # direction + focal key only meaningful on truly-relational rows
            dir_total += 1
            pred_dir = intent.get("direction") if pred_is_rel else None
            if pred_dir == row.get("direction"):
                dir_correct += 1

            key_total += 1
            pred_key = ra._resolve_focal_law(intent) if pred_is_rel else None
            if pred_key == _expected_focal_key(ra, row):
                key_correct += 1

    n = len(GOLD)
    report = {
        "n": n,
        "is_relational_acc": round(is_rel_correct / n, 4),
        "is_relational_recall": round(rel_tp / (rel_tp + rel_fn), 4) if (rel_tp + rel_fn) else 1.0,
        "direction_acc": round(dir_correct / dir_total, 4) if dir_total else 1.0,
        "focal_key_acc": round(key_correct / key_total, 4) if key_total else 1.0,
        "failures": failures,
    }
    write_report("relation_intent.json", report)
    with capsys.disabled():
        print("\n[relation-intent]", {k: v for k, v in report.items() if k != "failures"})
        if failures:
            print("[relation-intent] routing misses:", failures)

    assert report["is_relational_acc"] >= MIN_IS_RELATIONAL_ACC, report
    assert report["is_relational_recall"] >= MIN_IS_RELATIONAL_RECALL, report
    assert report["direction_acc"] >= MIN_DIRECTION_ACC, report
    assert report["focal_key_acc"] >= MIN_FOCAL_KEY_ACC, report
