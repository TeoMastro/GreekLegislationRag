"""Canonical-key builder for Greek law references.

Collapses inflected / abbreviated kind forms (ν., Ν., νόμος, νόμου, π.δ.,
προεδρικό διάταγμα, ...) into a small closed vocabulary, then produces a
stable canonical_key like 'Ν.4001/2011' that survives across queries and
extraction passes.
"""

from __future__ import annotations

import re
import unicodedata


KIND_LAW = "Ν."
KIND_PD = "Π.Δ."
KIND_PNP = "Π.Ν.Π."
KIND_YA = "Υ.Α."
KIND_AP = "Α.Π."
KIND_AN = "Α.Ν."   # αναγκαστικός νόμος — older decree-law; distinct from Ν.

CANONICAL_KINDS = {KIND_LAW, KIND_PD, KIND_PNP, KIND_YA, KIND_AP, KIND_AN}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)
    )


def _norm(s: str) -> str:
    return _strip_accents(s).lower().strip()


# Each entry maps a normalized (accent-folded, lowercased, whitespace-collapsed)
# kind form to the canonical kind string. Order doesn't matter — exact lookup.
_KIND_ALIASES: dict[str, str] = {
    "ν": KIND_LAW,
    "ν.": KIND_LAW,
    "νομος": KIND_LAW,
    "νομου": KIND_LAW,
    "νομο": KIND_LAW,
    "νομοι": KIND_LAW,
    "νομων": KIND_LAW,

    "π.δ.": KIND_PD,
    "π.δ": KIND_PD,
    "πδ": KIND_PD,
    "προεδρικο διαταγμα": KIND_PD,
    "προεδρικου διαταγματος": KIND_PD,

    "π.ν.π.": KIND_PNP,
    "π.ν.π": KIND_PNP,
    "πνπ": KIND_PNP,
    "πραξη νομοθετικου περιεχομενου": KIND_PNP,

    "υ.α.": KIND_YA,
    "υ.α": KIND_YA,
    "υα": KIND_YA,
    "υπουργικη αποφαση": KIND_YA,
    "υπουργικης αποφασης": KIND_YA,

    "α.π.": KIND_AP,
    "α.π": KIND_AP,
    "απ": KIND_AP,

    # Α.Ν. (αναγκαστικός νόμος) — require a dot or the full word so we don't
    # collide with the Greek conjunction "αν" ("if").
    "α.ν.": KIND_AN,
    "α.ν": KIND_AN,
    "αναγκαστικος νομος": KIND_AN,
    "αναγκαστικου νομου": KIND_AN,
    "αναγκαστικο νομο": KIND_AN,
}


_WS_RE = re.compile(r"\s+")


def canonicalize_kind(raw: str) -> str | None:
    """Map any inflected/abbreviated kind form to a canonical kind, or None."""
    if not raw:
        return None
    n = _WS_RE.sub(" ", _norm(raw)).strip(" .,")
    # Try as-is, then with trailing dot, then without dots — covers 'π δ', 'π.δ', 'π.δ.'
    candidates = {n, f"{n}.", n.replace(".", ""), n.replace(" ", "")}
    for c in candidates:
        if c in _KIND_ALIASES:
            return _KIND_ALIASES[c]
    return None


def canonical_key(kind: str, number: int, year: int) -> str:
    """Build the canonical_key string used as the law_nodes unique id.

    `kind` MUST already be one of CANONICAL_KINDS — call canonicalize_kind first.
    """
    if kind not in CANONICAL_KINDS:
        raise ValueError(f"kind {kind!r} is not canonical; canonicalize_kind first")
    return f"{kind}{int(number)}/{int(year)}"
