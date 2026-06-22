"""Citation extractor: regex pre-filter + LLM relation classifier.

Two-pass design so the LLM bill scales with the small fraction of chunks
that actually contain law references (~10–20% at ΦΕΚ corpus scale):

  1. `find_candidates(text)` — pure regex. Free, fast, returns the spans
     a chunk references. If empty, the chunk is skipped entirely.
  2. `classify_citations(text, candidates)` — gpt-5.4-mini in JSON mode.
     The regex candidates are passed in so the model only classifies the
     relation (τροποποιεί / καταργεί / αντικαθιστά / προστίθεται / αναφέρεται)
     instead of having to extract numbers itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from openai import OpenAI
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.citations.normalize import (
    CANONICAL_KINDS,
    canonical_key,
    canonicalize_kind,
)


_console = Console()


# Closed relation vocabulary. Anything outside this set is discarded.
RELATIONS = ("τροποποιεί", "καταργεί", "αντικαθιστά", "προστίθεται", "αναφέρεται")


# ----- Regex pre-filter -----------------------------------------------------

# Matches:
#   'Ν. 4001/2011', 'ν.4001/2011', 'νόμος 4001/2011', 'νόμου 4001/2011'
#   'Π.Δ. 56/2023', 'π. δ. 56/2023', 'προεδρικό διάταγμα 56/2023'
#   'Π.Ν.Π. 12/2024', 'υ.α. 99/2024'
# Kind alternation comes first; the (number)/(year) tail is shared.
_LAW_REF_RE = re.compile(
    r"""(?ix)
    \b
    (
        # Α.Ν. (αναγκαστικός νόμος) — must come BEFORE the bare ν. alternates,
        # otherwise the engine matches only the trailing 'ν.' and drops the 'α.'
        # prefix, miscategorising the citation as a plain Ν.
        α \. \s* ν \. ? \s* |
        αναγκαστικ \w* \s+ νόμ \w* \s+ |

        ν \. ? \s* |
        ν \s+ |
        νόμ \w* \s+ |          # νόμος/νόμου/νόμο/νόμοι/νόμων (all inflections)
        π \. \s* δ \. ? \s* |
        πδ \s* |
        προεδρικ \w* \s+ δι[αά]τ[αά]γμα \w* \s+ |
        π \. \s* ν \. \s* π \. ? \s* |
        πνπ \s* |
        πράξη \s+ νομοθετικού \s+ περιεχομένου \s+ |
        υ \. \s* α \. ? \s* |
        υα \s+ |
        υπουργικ[ήη]ς? \s+ απόφασης? \s+
    )
    (\d{1,5})
    \s* / \s*
    (\d{4})
    """
)


# Best-effort article number near a citation. Stand-alone — used only as a hint
# for the LLM; the model is the source of truth for whether an article applies.
_ARTICLE_RE = re.compile(
    r"(?i)\bάρθρ(?:ο|ου|α|ων)\s+(\d+[Α-Ωα-ωΆ-Ώάέήίόύώΐϊϋΰ]?)"
)


@dataclass(frozen=True)
class Candidate:
    kind: str              # canonical
    number: int
    year: int
    span_start: int        # char offset in source text
    span_end: int
    raw_kind: str          # what the regex matched, before canonicalization

    @property
    def canonical_key(self) -> str:
        return canonical_key(self.kind, self.number, self.year)


def find_candidates(text: str) -> list[Candidate]:
    """Return all unique (kind, number, year) law references in `text`."""
    if not text:
        return []
    seen: dict[str, Candidate] = {}
    for m in _LAW_REF_RE.finditer(text):
        raw_kind = m.group(1)
        canon = canonicalize_kind(raw_kind)
        if canon is None or canon not in CANONICAL_KINDS:
            continue
        try:
            number = int(m.group(2))
            year = int(m.group(3))
        except ValueError:
            continue
        cand = Candidate(
            kind=canon,
            number=number,
            year=year,
            span_start=m.start(),
            span_end=m.end(),
            raw_kind=raw_kind.strip(),
        )
        # Keep the first occurrence per canonical key (snippet context will
        # come from the chunk text anyway).
        seen.setdefault(cand.canonical_key, cand)
    return list(seen.values())


def has_citation(text: str) -> bool:
    """Fast check used to gate the LLM call. True iff regex finds anything."""
    if not text:
        return False
    return _LAW_REF_RE.search(text) is not None


# ----- LLM relation classifier ---------------------------------------------


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


_CLASSIFIER_SYSTEM = (
    "Είσαι ειδικός στην ελληνική νομοθεσία. Παίρνεις ένα απόσπασμα ΦΕΚ και μια "
    "λίστα από υποψήφιες παραπομπές που εντοπίστηκαν με regex. Για κάθε υποψήφια, "
    "αποφασίζεις αν είναι όντως παραπομπή και ταξινομείς τη σχέση από κλειστό "
    "σύνολο: 'τροποποιεί', 'καταργεί', 'αντικαθιστά', 'προστίθεται', 'αναφέρεται'. "
    "Επιστρέφεις ΜΟΝΟ έγκυρο JSON. Όταν δεν υπάρχει ρητό ρήμα σχέσης, επιστρέφεις "
    "'αναφέρεται'."
)


_CLASSIFIER_USER_TEMPLATE = """Απόσπασμα:
\"\"\"{text}\"\"\"

Υποψήφιες παραπομπές (kind, number, year):
{candidates_json}

Επέστρεψε JSON με κλειδί "citations" — λίστα αντικειμένων με τα πεδία:
- target_kind: ένα από {kinds}
- target_number: ακέραιος
- target_year: ακέραιος
- target_article: αριθμός άρθρου (string) ή null
- relation: ένα από {relations}
- snippet: μέχρι 200 χαρακτήρες από το απόσπασμα, γύρω από την παραπομπή
- confidence: 0.0–1.0

Παράδειγμα:
{{"citations": [
  {{"target_kind": "Ν.", "target_number": 4001, "target_year": 2011,
    "target_article": "5", "relation": "τροποποιεί",
    "snippet": "...τροποποιείται το άρθρο 5 του ν. 4001/2011...",
    "confidence": 0.95}}
]}}

Αν δεν υπάρχει καμία πραγματική παραπομπή, επέστρεψε {{"citations": []}}."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _classify_request(text: str, candidates: list[Candidate]) -> dict:
    cand_payload = [
        {"kind": c.kind, "number": c.number, "year": c.year}
        for c in candidates
    ]
    user = _CLASSIFIER_USER_TEMPLATE.format(
        text=text[:6000],  # cap context; chunk_tokens default is 512 so this is plenty
        candidates_json=json.dumps(cand_payload, ensure_ascii=False),
        kinds=list(CANONICAL_KINDS),
        relations=list(RELATIONS),
    )
    resp = _client().chat.completions.create(
        model=settings.citation_llm_model,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)


def classify_citations(
    text: str,
    candidates: list[Candidate],
) -> list[dict]:
    """Run the LLM relation classifier; return validated citation dicts.

    Each returned dict has keys:
      target_kind, target_number, target_year, target_article,
      target_canonical_key, relation, snippet, confidence
    Invalid entries (unknown kind / relation, out-of-range confidence, etc.)
    are dropped silently.
    """
    if not candidates:
        return []
    try:
        data = _classify_request(text, candidates)
    except Exception as e:
        _console.print(f"[yellow]citation classifier failed:[/yellow] {e}")
        return []

    items = data.get("citations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = canonicalize_kind(str(it.get("target_kind") or ""))
        if kind is None:
            continue
        try:
            number = int(it.get("target_number"))
            year = int(it.get("target_year"))
        except (TypeError, ValueError):
            continue
        relation = str(it.get("relation") or "").strip()
        if relation not in RELATIONS:
            # Map common variants the model might emit.
            base = relation.rstrip("ίαει")
            if base.startswith("τροποποι"):
                relation = "τροποποιεί"
            elif base.startswith("καταργ"):
                relation = "καταργεί"
            elif base.startswith("αντικαθιστ") or base.startswith("αντικαθίστ"):
                relation = "αντικαθιστά"
            elif base.startswith("προστιθ") or base.startswith("προστίθ") or base.startswith("προστεθ"):
                relation = "προστίθεται"
            else:
                relation = "αναφέρεται"
        try:
            confidence = float(it.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        target_article = it.get("target_article")
        if target_article is not None:
            target_article = str(target_article).strip() or None
        snippet = str(it.get("snippet") or "").strip()[:400] or text[:200]

        out.append(
            {
                "target_kind": kind,
                "target_number": number,
                "target_year": year,
                "target_canonical_key": canonical_key(kind, number, year),
                "target_article": target_article,
                "relation": relation,
                "snippet": snippet,
                "confidence": confidence,
            }
        )
    return out


def extract_from_chunk(text: str) -> list[dict]:
    """Convenience: regex-gate + classify. Returns [] if the chunk is empty."""
    candidates = find_candidates(text)
    if not candidates:
        return []
    return classify_citations(text, candidates)


def iter_extract_from_chunks(
    chunks: Iterable[tuple[int, str]],
) -> Iterable[tuple[int, list[dict]]]:
    """Yield (chunk_id, citations) pairs, skipping chunks with no regex hit."""
    for chunk_id, text in chunks:
        if not has_citation(text):
            continue
        cits = extract_from_chunk(text)
        if cits:
            yield chunk_id, cits
