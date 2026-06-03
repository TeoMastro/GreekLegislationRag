"""§6 Extraction-quality gate eval (deterministic) — guards the OCR text-quality
logic and documents the blind spot behind the always-OCR decision.

`assess_text_quality` decides whether an extracted text layer is good enough to
trust (vs. needing OCR). It gates on char COUNT, chars-per-page, and page COVERAGE
— but NOT on whether the text is actually readable Greek. That blind spot is why a
PDF with a dense-but-garbled text layer used to slip through (the resolved
garbled-text issue): the project now always-OCRs instead of trusting this gate.

These tests lock the gate's documented behaviour AND pin the blind spot, so nobody
re-introduces "trust the text layer when assess_text_quality passes" without
remembering it can't see garbling. The Greek-ratio assertions show the signal the
gate lacks (a cheap garbledness detector, if the gate is ever re-enabled).
"""

from src.ingestion.quality import (
    assess_text_quality,
    greek_char_count,
    latin_char_count,
)

# Defaults the gate reads (src/config.py): min_text_chars=500,
# min_chars_per_page=200, min_page_coverage=0.5.


def _chunks(*texts_pages):
    return [{"text": t, "pages": p} for t, p in texts_pages]


def test_empty_fails():
    ok, reason = assess_text_quality([], 0)
    assert ok is False and "no extractable text" in reason


def test_too_few_total_chars_fails():
    ok, reason = assess_text_quality(_chunks(("σύντομο κείμενο", [1])), 1)
    assert ok is False and "chars total" in reason


def test_low_chars_per_page_fails_scanned():
    # 600 chars over 100 pages -> 6 chars/page -> looks scanned.
    ok, reason = assess_text_quality(_chunks(("α" * 600, [1])), 100)
    assert ok is False and "chars/page" in reason


def test_low_page_coverage_fails():
    # Dense on one page, but only 1/4 pages have text.
    ok, reason = assess_text_quality(_chunks(("α" * 1000, [1])), 4)
    assert ok is False and "pages have text" in reason


def test_good_text_layer_passes():
    chunks = _chunks(("Το παρόν άρθρο ρυθμίζει " * 30, [1]),
                     ("τις λεπτομέρειες εφαρμογής " * 30, [2]))
    ok, reason = assess_text_quality(chunks, 2)
    assert ok is True and reason == ""


def test_blind_spot_garbled_but_dense_text_passes():
    # Mojibake / garbled OCR with plenty of chars and full coverage. The gate
    # has no readability check, so it PASSES — the documented blind spot that
    # motivated always-OCR. If this ever starts failing, the gate gained a
    # quality signal and the always-OCR rationale should be revisited.
    garbled = "Ã¢â‚¬Å¡Ã‚Â¬Ã¯Â¿Â½ " * 80  # >500 chars, no real Greek
    ok, _ = assess_text_quality(_chunks((garbled, [1])), 1)
    assert ok is True


# --------------------------------------------------------------------------- #
# The Greek-ratio signal the gate lacks (cheap garbledness detector).
# --------------------------------------------------------------------------- #

def _greek_ratio(text: str) -> float:
    g, l = greek_char_count(text), latin_char_count(text)
    denom = g + l
    return g / denom if denom else 0.0


def test_greek_ratio_separates_clean_from_garbled():
    clean = "Με την παρούσα απόφαση ρυθμίζονται τα θέματα εφαρμογής του νόμου."
    garbled = "Ã¢â‚¬Å¡Ã‚Â¬ a b c d e f g h"
    assert _greek_ratio(clean) > 0.8
    assert _greek_ratio(garbled) < 0.2
