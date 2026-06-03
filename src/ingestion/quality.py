"""Dependency-light text-quality heuristics.

Kept separate from ingest.py (which imports the heavy docling / PDF / embedding
stack) so these pure functions can be imported — and unit-tested — cheaply.

NOTE: `assess_text_quality` judges QUANTITY (char count, chars/page, coverage),
not READABILITY. It cannot tell clean Greek from garbled mojibake — the blind spot
that motivated always-OCR. `greek_char_count`/`latin_char_count` provide the
readability signal it lacks (see evals/component/test_text_quality.py).
"""

from src.config import settings


def greek_char_count(text: str) -> int:
    return sum(1 for c in text if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")


def latin_char_count(text: str) -> int:
    return sum(1 for c in text if "a" <= c.lower() <= "z")


def assess_text_quality(
    chunks: list[dict], total_pages: int
) -> tuple[bool, str]:
    if not chunks:
        return False, "no extractable text"
    total_chars = sum(len(c["text"].strip()) for c in chunks)
    if total_chars < settings.min_text_chars:
        return False, f"only {total_chars} chars total"
    if total_pages > 0:
        pages_with_text = len({p for c in chunks for p in (c.get("pages") or [])})
        chars_per_page = total_chars / total_pages
        coverage = pages_with_text / total_pages
        if chars_per_page < settings.min_chars_per_page:
            return False, (
                f"{chars_per_page:.0f} chars/page over {total_pages} pages "
                f"(likely scanned)"
            )
        if coverage < settings.min_page_coverage:
            return False, (
                f"only {pages_with_text}/{total_pages} pages have text "
                f"(likely scanned)"
            )
    return True, ""
