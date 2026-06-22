"""§1.4 Answer citation-format eval — guards _strip_invalid_citations.

The combiner enforces that every [n] in an answer points at a real source. This
locks that contract: in-range citations survive, out-of-range are stripped, and
punctuation spacing is repaired. Pure string logic, no API.
"""

import pytest

from src.rag.agents.combiner_agent import _strip_invalid_citations


@pytest.mark.parametrize(
    "answer, num_sources, expected",
    [
        ("Ο νόμος ισχύει [1].", 3, "Ο νόμος ισχύει [1]."),
        ("Πρώτο [1], δεύτερο [2], τρίτο [3].", 3, "Πρώτο [1], δεύτερο [2], τρίτο [3]."),
        # out-of-range stripped, dangling space before punctuation repaired
        ("Ισχύει [5].", 3, "Ισχύει."),
        # stripping a mid-sentence citation also removes its leading space, so
        # no double space is left behind (FINDINGS #6).
        ("Πηγές [1] και [9] το λένε.", 3, "Πηγές [1] και το λένε."),
        # zero is out of range
        ("Δες [0].", 3, "Δες."),
        # no citations at all -> untouched
        ("Δεν βρέθηκε σχετική πληροφορία.", 0, "Δεν βρέθηκε σχετική πληροφορία."),
    ],
)
def test_strip_invalid_citations(answer, num_sources, expected):
    assert _strip_invalid_citations(answer, num_sources) == expected


def test_all_citations_stripped_when_no_sources():
    # With 0 sources every bracketed ref is invalid and must go.
    assert "[" not in _strip_invalid_citations("Α [1] Β [2].", 0)
