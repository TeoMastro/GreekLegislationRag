"""§1.2 Regex candidate-finder eval — the citation graph's recall ceiling.

Chunks with no regex hit never reach the LLM classifier, so a false negative here
is permanent: that edge can never enter the graph. We therefore gate on RECALL
(primary) and track false positives loosely (the LLM filters them downstream).

The thresholds below are a *regression baseline* locked to current behaviour — not
an aspiration. When you improve the regex, raise them. When you add real OCR'd
chunks that expose a new miss, the per-format breakdown in the emitted report tells
you exactly which shape regressed. Run `pytest -s` to print the breakdown inline.
"""

from collections import defaultdict

from evals._util import load_jsonl, write_report
from src.citations.extractor import find_candidates

DATA = load_jsonl("regex_candidates.jsonl")
# `known_gap` entries are citation shapes the regex provably MISSES today (e.g.
# nominative 'νόμος', genitive 'προεδρικού διατάγματος'). They're kept in the
# dataset so the gap stays visible and the recall-including-gaps number is honest,
# but they don't gate CI — otherwise the suite could never go green. Remove the
# flag (and the gap closes) when the regex is improved.
POSITIVES = [r for r in DATA if r["expected_keys"] and not r.get("known_gap")]
KNOWN_GAPS = [r for r in DATA if r["expected_keys"] and r.get("known_gap")]
NEGATIVES = [r for r in DATA if not r["expected_keys"]]

# Regression baseline. Recall over SUPPORTED shapes must not drop below this;
# FP rate must not exceed it.
MIN_RECALL = 1.0
MAX_FP_RATE = 0.0


def _found_keys(text: str) -> set[str]:
    return {c.canonical_key for c in find_candidates(text)}


def _compute():
    per_format = defaultdict(lambda: {"expected": 0, "found": 0})
    total_expected = 0
    total_found = 0
    misses = []
    for row in POSITIVES:
        got = _found_keys(row["text"])
        want = set(row["expected_keys"])
        hit = want & got
        per_format[row["format"]]["expected"] += len(want)
        per_format[row["format"]]["found"] += len(hit)
        total_expected += len(want)
        total_found += len(hit)
        for k in want - got:
            misses.append({"id": row["id"], "format": row["format"], "missing": k})

    fp = []
    for row in NEGATIVES:
        spurious = _found_keys(row["text"])
        if spurious:
            fp.append({"id": row["id"], "spurious": sorted(spurious)})

    recall = total_found / total_expected if total_expected else 1.0
    fp_rate = len(fp) / len(NEGATIVES) if NEGATIVES else 0.0

    # Honest overall recall, counting the known-gap shapes the regex misses.
    gap_expected = sum(len(r["expected_keys"]) for r in KNOWN_GAPS)
    gap_found = sum(len(set(r["expected_keys"]) & _found_keys(r["text"])) for r in KNOWN_GAPS)
    overall_expected = total_expected + gap_expected
    overall_found = total_found + gap_found

    return {
        "recall_supported": round(recall, 4),
        "recall_overall": round(overall_found / overall_expected, 4) if overall_expected else 1.0,
        "expected": total_expected,
        "found": total_found,
        "fp_rate": round(fp_rate, 4),
        "false_positives": fp,
        "misses": misses,
        "known_gaps": [{"id": r["id"], "why": r["known_gap"]} for r in KNOWN_GAPS],
        "per_format": {
            fmt: round(v["found"] / v["expected"], 4) if v["expected"] else None
            for fmt, v in sorted(per_format.items())
        },
    }


def test_regex_recall_and_precision(capsys):
    report = _compute()
    write_report("regex_candidates.json", report)

    with capsys.disabled():
        print(
            "\n[regex] recall_supported={recall_supported} "
            "recall_overall={recall_overall} fp_rate={fp_rate}".format(**report)
        )
        print("[regex] per-format recall:", report["per_format"])
        if report["misses"]:
            print("[regex] UNEXPECTED MISSES:", report["misses"])
        if report["known_gaps"]:
            print("[regex] KNOWN GAPS (tracked, not gating):", report["known_gaps"])
        if report["false_positives"]:
            print("[regex] FALSE POSITIVES:", report["false_positives"])

    assert report["recall_supported"] >= MIN_RECALL, (
        f"regex recall regressed on supported shapes: {report['misses']}"
    )
    assert report["fp_rate"] <= MAX_FP_RATE, f"new false positives: {report['false_positives']}"
