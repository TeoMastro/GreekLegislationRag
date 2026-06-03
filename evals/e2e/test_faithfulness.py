"""§5.1 Faithfulness + §5.4/§5.5 format/citation checks (online, E2E).

Runs the full multi-agent pipeline on answerable questions and judges whether the
answer's claims are grounded in the sources it retrieved — the core safety check
for a legal RAG (a hallucinated law number is the worst failure mode). No reference
answer is needed: faithfulness is measured against the pipeline's OWN sources, and
an honest abstention counts as fully faithful.

Folded in (cheap, deterministic):
  - language: the answer is in Greek
  - citations: every [n] marker is within the source count (regression guard on
    CombinerAgent._strip_invalid_citations)

Online tier — needs OpenAI + Supabase + a populated corpus. Run with:
    RUN_ONLINE_EVALS=1 pytest evals/e2e/test_faithfulness.py -s
"""

from __future__ import annotations

import os
import re

import pytest

from evals._util import DATASETS_DIR, write_report

RUN = os.environ.get("RUN_ONLINE_EVALS") == "1"
DATASET = DATASETS_DIR / "synthetic_queries.jsonl"
N = int(os.environ.get("E2E_N", "12"))  # pipeline runs are slow; keep small

MIN_FAITHFULNESS_MEAN = 0.80   # calibrate on first run
MIN_GREEK_RATE = 1.0
MIN_CITATION_VALID_RATE = 1.0

_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
_CITE_RE = re.compile(r"\[(\d+)\]")

pytestmark = pytest.mark.online


def _is_greek(text: str) -> bool:
    return bool(_GREEK_RE.search(text or ""))


def _citations_valid(answer: str, n_sources: int) -> bool:
    return all(1 <= int(m) <= n_sources for m in _CITE_RE.findall(answer or ""))


@pytest.mark.skipif(not RUN, reason="online eval; set RUN_ONLINE_EVALS=1")
@pytest.mark.skipif(not DATASET.exists(), reason="run synthetic_gen.py first")
def test_faithfulness(capsys):
    import json

    from evals.e2e.judges import judge_faithfulness
    from src.rag.graph import run_multi_agent_query

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()][:N]

    faith_scores = []
    greek_ok = cite_ok = 0
    low = []  # below-threshold answers for inspection
    evaluated = 0

    for i, r in enumerate(rows):
        out = run_multi_agent_query(r["question"], session_id=f"eval-faith-{i}")
        answer = out.get("answer") or ""
        sources = out.get("sources") or []
        evaluated += 1

        if _is_greek(answer):
            greek_ok += 1
        if _citations_valid(answer, len(sources)):
            cite_ok += 1

        context = "\n\n---\n\n".join(
            f"[{j+1}] {getattr(s, 'page_content', '') or ''}" for j, s in enumerate(sources)
        )
        verdict = judge_faithfulness(r["question"], answer, context)
        f = verdict.get("faithfulness")
        if isinstance(f, (int, float)):
            faith_scores.append(float(f))
            if f < MIN_FAITHFULNESS_MEAN:
                low.append({"q": r["question"][:80], "faithfulness": f,
                            "unsupported": verdict.get("unsupported")})

    n = evaluated or 1
    report = {
        "n": evaluated,
        "faithfulness_mean": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        "faithfulness_min": round(min(faith_scores), 4) if faith_scores else None,
        "greek_rate": round(greek_ok / n, 4),
        "citation_valid_rate": round(cite_ok / n, 4),
        "low_faithfulness": low,
    }
    write_report("faithfulness.json", report)
    with capsys.disabled():
        print("\n[faithfulness]", {k: v for k, v in report.items() if k != "low_faithfulness"})
        if low:
            print("[faithfulness] below threshold:", low)

    assert report["greek_rate"] >= MIN_GREEK_RATE, report
    assert report["citation_valid_rate"] >= MIN_CITATION_VALID_RATE, report
    assert report["faithfulness_mean"] is not None and \
        report["faithfulness_mean"] >= MIN_FAITHFULNESS_MEAN, report
