"""§1.5 Listing-table parser eval — guards against scraper markdown drift.

ListingAgent's structured-intent retrieval depends entirely on parsing the
per-year `listing-items-YYYY.md` tables the scraper emits. A column reorder, a
date-format change, or a markdown-link shape change silently breaks known-item
retrieval. This locks the parse of a representative table.
"""

from datetime import date
from pathlib import Path

from src.rag.agents.listing_agent import _parse_listing_file

FIXTURE = Path(__file__).parent / "fixtures" / "listing-items-sample.md"
ROWS = _parse_listing_file(FIXTURE)


def test_row_count_skips_header_and_separator():
    assert len(ROWS) == 3


def test_first_row_fields():
    r = ROWS[0]
    assert r["kind"] == "Ν."
    assert r["number"] == 5167
    assert r["description"] == "Φορολογικές ρυθμίσεις και λοιπές διατάξεις"
    assert r["fek_series"] == "Α"
    assert r["fek_number"] == 90
    assert r["fek_year"] == 2024
    assert r["fek_date"] == date(2024, 5, 15)
    assert r["pages_count"] == 42
    assert r["pdf_basename"] == "20240100090.pdf"


def test_pdf_path_falls_back_to_download_when_no_bookmark():
    # Row 3 has an empty Σελιδοδείκτης cell -> pdf_path must use the Λήψη link.
    r = ROWS[2]
    assert r["bookmark_path"] == ""
    assert r["pdf_path"] == "2020/20200100184.pdf"
    assert r["pdf_basename"] == "20200100184.pdf"


def test_kinds_and_basenames():
    assert [r["kind"] for r in ROWS] == ["Ν.", "Π.Δ.", "Ν."]
    assert [r["pdf_basename"] for r in ROWS] == [
        "20240100090.pdf",
        "20230100104.pdf",
        "20200100184.pdf",
    ]
