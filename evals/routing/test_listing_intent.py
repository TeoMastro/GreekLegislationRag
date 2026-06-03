"""§4.2 ListingAgent intent eval (online) — structured-filter extraction.

Scored as exact-match over the scalar filter fields each gold row declares in
`expected` (a field declared null MUST come back null — over-extraction is as bad
as under-extraction because it wrongly narrows the candidate set). `keywords_any`
is a loose recall check on keyword extraction.

Online tier — needs OpenAI. Run with:
    RUN_ONLINE_EVALS=1 pytest evals/routing/test_listing_intent.py -s
"""

from __future__ import annotations

import os

import pytest

from evals._util import load_jsonl, write_report
from src.rag.agents.listing_agent import _normalize

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
GOLD = load_jsonl("listing_intent.jsonl")

# Regression floors — calibrate on first clean run.
MIN_FIELD_ACC = 0.85
MIN_KEYWORD_RECALL = 0.80

pytestmark = pytest.mark.online


def _keywords_present(pred_keywords, wanted) -> bool:
    """True if at least one wanted keyword appears (accent-folded) among the
    predicted keywords."""
    pred_norm = {_normalize(k) for k in (pred_keywords or []) if isinstance(k, str)}
    return any(_normalize(w) in pred_norm for w in wanted)


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
def test_listing_intent(capsys):
    from src.rag.agents.listing_agent import get_listing_agent

    agent = get_listing_agent()

    field_correct = field_total = 0
    kw_correct = kw_total = 0
    failures = []

    for row in GOLD:
        intent = agent._extract_intent(row["query"])

        for field, want in row["expected"].items():
            field_total += 1
            got = intent.get(field)
            if got == want:
                field_correct += 1
            else:
                failures.append({"query": row["query"], "field": field, "want": want, "got": got})

        wanted_kw = row.get("keywords_any")
        if wanted_kw:
            kw_total += 1
            if _keywords_present(intent.get("keywords"), wanted_kw):
                kw_correct += 1
            else:
                failures.append({"query": row["query"], "field": "keywords", "want_any": wanted_kw,
                                 "got": intent.get("keywords")})

    report = {
        "n": len(GOLD),
        "field_acc": round(field_correct / field_total, 4) if field_total else 1.0,
        "fields_checked": field_total,
        "keyword_recall": round(kw_correct / kw_total, 4) if kw_total else 1.0,
        "failures": failures,
    }
    write_report("listing_intent.json", report)
    with capsys.disabled():
        print("\n[listing-intent]", {k: v for k, v in report.items() if k != "failures"})
        if failures:
            print("[listing-intent] field misses:", failures)

    assert report["field_acc"] >= MIN_FIELD_ACC, report
    assert report["keyword_recall"] >= MIN_KEYWORD_RECALL, report
