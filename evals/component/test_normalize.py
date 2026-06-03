"""§1.1 Canonicalization eval — pure functions, hard 100%-exact invariant.

The canonical_key is the citation graph's join key; any drift silently splits a
law into two nodes (or merges Α.Ν. into Ν.), so this is pass/fail, not a metric.
"""

import pytest

from evals._util import load_jsonl
from src.citations.normalize import canonical_key, canonicalize_kind

GOLDEN = load_jsonl("normalize_golden.jsonl")


@pytest.mark.parametrize("row", GOLDEN, ids=[r["raw"] or "<empty>" for r in GOLDEN])
def test_canonicalize_kind(row):
    assert canonicalize_kind(row["raw"]) == row["expected_kind"]


@pytest.mark.parametrize(
    "row",
    [r for r in GOLDEN if r.get("expected_key")],
    ids=[r["expected_key"] for r in GOLDEN if r.get("expected_key")],
)
def test_canonical_key(row):
    kind = canonicalize_kind(row["raw"])
    assert kind == row["expected_kind"]
    assert canonical_key(kind, row["number"], row["year"]) == row["expected_key"]
