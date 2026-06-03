"""§4 Routing logic eval (deterministic part) — the pure intent post-processing.

The LLM intent *extraction* is online (see evals/routing/), but everything the
agents do with an intent dict is pure Python and gates here for free:

  - RelationAgent._resolve_focal_law / _validate_relations — turn a parsed intent
    into a canonical graph key + a validated relation filter. A bug here sends the
    wrong key to the citation RPCs (wrong/empty graph answer).
  - ListingAgent._row_passes_hard_filters / _select_candidates — the structured
    filter that decides WHICH ΦΕΚ a "Ν. 5263" / "ΦΕΚ Α 238/2025" / date-range query
    retrieves. A bug here silently returns the wrong document.

These take intent dicts as input, so no network is needed.
"""

from datetime import date

import pytest

from src.rag.agents import listing_agent as la
from src.rag.agents.listing_agent import (
    ListingRow,
    _query_keywords,
    _row_passes_hard_filters,
    _select_candidates,
)
from src.rag.agents.relation_agent import get_relation_agent

RA = get_relation_agent()


# --------------------------------------------------------------------------- #
# RelationAgent: focal-law resolution + relation validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "intent, expected",
    [
        ({"source_law": {"kind": "Ν.", "number": 5167, "year": 2024}}, "Ν.5167/2024"),
        ({"source_law": {"kind": "ν.", "number": 4001, "year": 2011}}, "Ν.4001/2011"),
        ({"source_law": {"kind": "α.ν.", "number": 1846, "year": 1951}}, "Α.Ν.1846/1951"),
        ({"source_law": {"kind": "Π.Δ.", "number": 18, "year": 1989}}, "Π.Δ.18/1989"),
        # unresolvable / malformed -> None (agent then emits empty results)
        ({"source_law": {"kind": "foo", "number": 1, "year": 2}}, None),
        ({"source_law": {"kind": "Ν.", "number": "x", "year": 2024}}, None),
        ({"source_law": None}, None),
        ({}, None),
    ],
)
def test_resolve_focal_law(intent, expected):
    assert RA._resolve_focal_law(intent) == expected


@pytest.mark.parametrize(
    "intent, expected",
    [
        ({"relations": ["τροποποιεί"]}, ["τροποποιεί"]),
        ({"relations": ["τροποποιεί", "garbage", "καταργεί"]}, ["τροποποιεί", "καταργεί"]),
        ({"relations": ["garbage"]}, None),   # nothing valid -> None (= all relations)
        ({"relations": None}, None),
        ({"relations": "τροποποιεί"}, None),   # not a list -> None
        ({}, None),
    ],
)
def test_validate_relations(intent, expected):
    assert RA._validate_relations(intent) == expected


# --------------------------------------------------------------------------- #
# ListingAgent: hard structured filters
# --------------------------------------------------------------------------- #

def _row(**kw) -> ListingRow:
    base = dict(
        kind="Ν.", number=5167, description="φορολογικές ρυθμίσεις",
        fek_title_raw="Α 90/2024", fek_series="Α", fek_number=90, fek_year=2024,
        fek_date=date(2024, 5, 15), pages_count=42, pdf_basename="20240100090.pdf",
    )
    base.update(kw)
    return ListingRow(**base)


@pytest.mark.parametrize(
    "intent, year, expected",
    [
        ({}, 2024, True),
        ({}, 2023, False),                         # year mismatch via fek_date
        ({"kind": "Ν."}, None, True),
        ({"kind": "Π.Δ."}, None, False),
        ({"number": 5167}, None, True),
        ({"number": 9999}, None, False),
        ({"fek_series": "Α", "fek_number": 90, "fek_year": 2024}, None, True),
        ({"fek_number": 91}, None, False),
        ({"min_pages": 40}, None, True),
        ({"min_pages": 100}, None, False),
        ({"max_pages": 50}, None, True),
        ({"max_pages": 10}, None, False),
        ({"date_from": "2024-01-01", "date_to": "2024-12-31"}, None, True),
        ({"date_from": "2024-06-01"}, None, False),  # row is 2024-05-15
    ],
)
def test_row_passes_hard_filters(intent, year, expected):
    assert _row_passes_hard_filters(_row(), intent, year) is expected


def test_year_falls_back_to_fek_year_when_no_date():
    # No fek_date -> the year gate compares against fek_year instead.
    r = _row(fek_date=None, fek_year=2024)
    assert _row_passes_hard_filters(r, {}, 2024) is True
    assert _row_passes_hard_filters(r, {}, 2023) is False


def test_query_keywords_strips_stopwords():
    kws = _query_keywords("ποιος νόμος για τη φορολογία ακινήτων", None)
    assert "φορολογία" in kws and "ακινήτων" in kws
    # stopwords / the kind word 'νομος' are removed
    assert all(k.lower() not in {"ποιος", "για", "τη", "νομος"} for k in kws)


# --------------------------------------------------------------------------- #
# ListingAgent: candidate selection over a controlled listing set
# --------------------------------------------------------------------------- #

_LISTINGS = [
    _row(number=5167, description="φορολογικές ρυθμίσεις ακινήτων", pdf_basename="a.pdf"),
    _row(number=4727, kind="Ν.", description="ψηφιακή διακυβέρνηση",
         fek_series="Α", fek_number=184, fek_year=2020,
         fek_date=date(2020, 9, 23), pages_count=210, pdf_basename="b.pdf"),
    _row(number=56, kind="Π.Δ.", description="οργανισμός υπουργείου υγείας",
         fek_series="Α", fek_number=104, fek_year=2023,
         fek_date=date(2023, 6, 20), pages_count=12, pdf_basename="c.pdf"),
]


@pytest.fixture
def listings(monkeypatch):
    monkeypatch.setattr(la, "_load_listings", lambda: list(_LISTINGS))


def test_select_by_exact_number(listings):
    out = _select_candidates("Ν. 5167", {"kind": "Ν.", "number": 5167}, None, top_n=5)
    assert [r["pdf_basename"] for r in out] == ["a.pdf"]


def test_select_by_keyword_ranks_relevant_first(listings):
    out = _select_candidates("φορολογία ακινήτων", {"keywords": ["φορολογία", "ακινήτων"]}, None, 5)
    assert out and out[0]["pdf_basename"] == "a.pdf"


def test_select_with_year_constraint(listings):
    out = _select_candidates("νόμοι", {}, 2020, top_n=5)
    assert [r["pdf_basename"] for r in out] == ["b.pdf"]


def test_select_no_match_returns_empty(listings):
    out = _select_candidates("τίποτα σχετικό εδώ", {"keywords": ["ανύπαρκτο"]}, None, 5)
    assert out == []
