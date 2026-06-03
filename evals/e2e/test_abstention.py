"""§5.2 Abstention / refusal calibration (online, E2E).

For a legal assistant, confidently answering a question the corpus can't support
is the most dangerous failure. This measures:
  - abstention_rate on UNANSWERABLE queries (should be high) — the retriever always
    returns top-k chunks, so the combiner must recognise they don't answer the
    question and say so rather than fabricate.
  - false_abstention_rate on a few ANSWERABLE controls (should be low) — guards
    against an over-cautious assistant that refuses real questions.

Online tier — needs OpenAI + Supabase. Run with:
    RUN_ONLINE_EVALS=1 pytest evals/e2e/test_abstention.py -s
"""

from __future__ import annotations

import json
import os

import pytest

from evals._util import DATASETS_DIR, load_jsonl, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
UNANSWERABLE = load_jsonl("unanswerable.jsonl")
CONTROL_SRC = DATASETS_DIR / "synthetic_queries.jsonl"
N_CONTROLS = int(os.environ.get("E2E_CONTROLS", "4"))

MIN_ABSTENTION_RATE = 0.80          # unanswerable -> should abstain
MAX_FALSE_ABSTENTION_RATE = 0.25    # answerable -> should NOT abstain

# Deterministic refusal strings the CombinerAgent emits directly.
_HARD_REFUSALS = (
    "Δεν βρέθηκαν σχετικά αποσπάσματα",
    "δεν ήταν δυνατή η ανάκτηση",
)

pytestmark = pytest.mark.online


def _abstained(query: str, answer: str) -> bool:
    if any(s in (answer or "") for s in _HARD_REFUSALS):
        return True
    from evals.e2e.judges import judge_abstention

    return bool(judge_abstention(query, answer).get("abstained"))


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
def test_abstention(capsys):
    from src.rag.graph import run_multi_agent_query

    # Unanswerable: count abstentions.
    abstained = 0
    leaked = []  # unanswerable queries that got a substantive answer
    for i, r in enumerate(UNANSWERABLE):
        out = run_multi_agent_query(r["query"], session_id=f"eval-abs-{i}")
        ans = out.get("answer") or ""
        if _abstained(r["query"], ans):
            abstained += 1
        else:
            leaked.append({"query": r["query"], "reason": r["reason"], "answer": ans[:160]})

    # Answerable controls: count FALSE abstentions.
    false_abst = 0
    controls = []
    if CONTROL_SRC.exists():
        controls = [json.loads(l) for l in CONTROL_SRC.read_text(encoding="utf-8").splitlines()
                    if l.strip()][:N_CONTROLS]
    for i, r in enumerate(controls):
        out = run_multi_agent_query(r["question"], session_id=f"eval-absctrl-{i}")
        ans = out.get("answer") or ""
        if _abstained(r["question"], ans):
            false_abst += 1

    nu = len(UNANSWERABLE) or 1
    nc = len(controls) or 1
    report = {
        "n_unanswerable": len(UNANSWERABLE),
        "abstention_rate": round(abstained / nu, 4),
        "n_controls": len(controls),
        "false_abstention_rate": round(false_abst / nc, 4) if controls else None,
        "leaked": leaked,
    }
    write_report("abstention.json", report)
    with capsys.disabled():
        print("\n[abstention]", {k: v for k, v in report.items() if k != "leaked"})
        if leaked:
            print("[abstention] FABRICATED on unanswerable:", leaked)

    assert report["abstention_rate"] >= MIN_ABSTENTION_RATE, report
    if controls:
        assert report["false_abstention_rate"] <= MAX_FALSE_ABSTENTION_RATE, report
