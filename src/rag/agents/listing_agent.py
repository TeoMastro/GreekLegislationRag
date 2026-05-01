import json
import os
import re
import unicodedata
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.config import settings
from src.rag.agents.base_agent import BaseAgent
from src.rag.retriever import dict_to_document
from src.retrieval.store import hybrid_search


_console = Console()


_EXPECTED_HEADERS = [
    "Είδος",
    "Αριθμός",
    "Περιγραφή",
    "Τίτλος ΦΕΚ",
    "Ημερομηνία ΦΕΚ",
    "Σελίδες",
    "Λήψη",
    "Σελιδοδείκτης",
]

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_FEK_TITLE_RE = re.compile(r"^\s*([Ͱ-Ͽἀ-῿]+)\s+(\d+)\s*/\s*(\d{4})\s*$")


class ListingRow(TypedDict, total=False):
    kind: str
    number: int | None
    description: str
    fek_title_raw: str
    fek_series: str | None
    fek_number: int | None
    fek_year: int | None
    fek_date: date | None
    pages_count: int | None
    download_path: str
    bookmark_path: str
    pdf_path: str
    pdf_basename: str


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)
    )


def _normalize(s: str) -> str:
    return _strip_accents(s).lower()


def _parse_md_link(cell: str) -> tuple[str, str]:
    m = _MD_LINK_RE.search(cell)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", cell.strip()


def _parse_int(cell: str) -> int | None:
    cell = cell.strip()
    if not cell:
        return None
    try:
        return int(cell)
    except ValueError:
        return None


def _parse_date(cell: str) -> date | None:
    cell = cell.strip()
    if not cell:
        return None
    try:
        return datetime.strptime(cell, "%d-%m-%Y").date()
    except ValueError:
        return None


def _parse_fek_title(cell: str) -> tuple[str, int | None, int | None]:
    m = _FEK_TITLE_RE.match(cell.strip())
    if not m:
        return "", None, None
    return m.group(1), _parse_int(m.group(2)), _parse_int(m.group(3))


def _parse_listing_file(path: Path) -> list[ListingRow]:
    rows: list[ListingRow] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 7:
            continue
        if cells[0] == "Είδος" or set(cells[0]) <= {"-", " ", ":"}:
            continue
        while len(cells) < 8:
            cells.append("")

        download_label, download_path = _parse_md_link(cells[6])
        _, bookmark_path = _parse_md_link(cells[7])
        pdf_path = bookmark_path or download_path
        if not pdf_path:
            continue

        fek_title_raw = cells[3]
        fek_series, fek_number, fek_year = _parse_fek_title(fek_title_raw)

        rows.append(
            ListingRow(
                kind=cells[0],
                number=_parse_int(cells[1]),
                description=cells[2],
                fek_title_raw=fek_title_raw,
                fek_series=fek_series or None,
                fek_number=fek_number,
                fek_year=fek_year,
                fek_date=_parse_date(cells[4]),
                pages_count=_parse_int(cells[5]),
                download_path=download_path,
                bookmark_path=bookmark_path,
                pdf_path=pdf_path,
                pdf_basename=os.path.basename(pdf_path),
            )
        )
    return rows


@lru_cache(maxsize=1)
def _load_listings() -> list[ListingRow]:
    rows: list[ListingRow] = []
    for md in sorted(settings.downloads_dir.glob("listing-items-*.md")):
        rows.extend(_parse_listing_file(md))
    return rows


_INTENT_SYSTEM = (
    "Είσαι βοηθός που εξάγει δομημένα φίλτρα από ερωτήματα χρήστη πάνω στην ελληνική "
    "νομοθεσία. Επιστρέφεις ΜΟΝΟ έγκυρο JSON με τα κλειδιά που αντιστοιχούν σε στήλες "
    "πίνακα ΦΕΚ. Όπου δεν εφαρμόζεται φίλτρο, χρησιμοποίησε null."
)

_INTENT_USER_TEMPLATE = """Σήμερα είναι {today}. Οι διαθέσιμες στήλες είναι:
- Είδος (π.χ. "Ν.", "Π.Δ.", "Π.Ν.Π.")
- Αριθμός (αριθμός νόμου, ακέραιος)
- Περιγραφή (ελεύθερο κείμενο)
- Τίτλος ΦΕΚ (π.χ. "Α 238/2025": σειρά / αριθμός / έτος)
- Ημερομηνία ΦΕΚ
- Σελίδες (ακέραιος)

Από το ερώτημα: {query}

Επέστρεψε JSON με τα παρακάτω κλειδιά (όλα προαιρετικά, χρησιμοποίησε null όπου δεν εφαρμόζεται):
{{
  "date_from": "YYYY-MM-DD" ή null,
  "date_to": "YYYY-MM-DD" ή null,
  "year": ακέραιος ή null,
  "kind": "Ν." | "Π.Δ." | "Π.Ν.Π." | άλλο ή null,
  "number": ακέραιος ή null,
  "fek_series": "Α" | "Β" | "Γ" | άλλο ή null,
  "fek_number": ακέραιος ή null,
  "fek_year": ακέραιος ή null,
  "min_pages": ακέραιος ή null,
  "max_pages": ακέραιος ή null,
  "keywords": [λίστα λέξεων-κλειδιά για ταίριασμα στην Περιγραφή/Τίτλο]
}}

Παραδείγματα:
- "Ν. 5263" → {{"kind":"Ν.","number":5263,"keywords":[]}}
- "ΦΕΚ Α 238/2025" → {{"fek_series":"Α","fek_number":238,"fek_year":2025,"keywords":[]}}
- "νόμοι μεταξύ Αύγουστο και Σεπτέμβριο 2025" → {{"date_from":"2025-08-01","date_to":"2025-09-30","keywords":[]}}
- "νόμοι του 2024 με πάνω από 100 σελίδες" → {{"year":2024,"min_pages":100,"keywords":[]}}
- "νόμος για την τεχνητή νοημοσύνη" → {{"keywords":["τεχνητή","νοημοσύνη"]}}
"""


def _row_passes_hard_filters(row: ListingRow, intent: dict, year: int | None) -> bool:
    yr = year if year is not None else intent.get("year")
    if yr is not None and row.get("fek_date") and row["fek_date"].year != yr:
        return False
    if yr is not None and row.get("fek_date") is None and row.get("fek_year") != yr:
        return False

    df = intent.get("date_from")
    dt = intent.get("date_to")
    if (df or dt) and row.get("fek_date"):
        if df:
            try:
                if row["fek_date"] < datetime.strptime(df, "%Y-%m-%d").date():
                    return False
            except ValueError:
                pass
        if dt:
            try:
                if row["fek_date"] > datetime.strptime(dt, "%Y-%m-%d").date():
                    return False
            except ValueError:
                pass

    kind = intent.get("kind")
    if kind and _normalize(row.get("kind", "")) != _normalize(kind):
        return False

    number = intent.get("number")
    if number is not None and row.get("number") != number:
        return False

    fek_series = intent.get("fek_series")
    if fek_series and _normalize(row.get("fek_series") or "") != _normalize(fek_series):
        return False

    fek_number = intent.get("fek_number")
    if fek_number is not None and row.get("fek_number") != fek_number:
        return False

    fek_year = intent.get("fek_year")
    if fek_year is not None and row.get("fek_year") != fek_year:
        return False

    min_pages = intent.get("min_pages")
    if min_pages is not None:
        if row.get("pages_count") is None or row["pages_count"] < min_pages:
            return False

    max_pages = intent.get("max_pages")
    if max_pages is not None:
        if row.get("pages_count") is None or row["pages_count"] > max_pages:
            return False

    return True


_GREEK_STOPWORDS = {
    "ο","η","το","οι","τα","του","της","των","τον","την","και","ή","να","σε",
    "με","για","από","στο","στη","στην","στον","στους","στις","ένας","μία","ένα",
    "που","ποιος","ποια","ποιο","ποιοι","ποιες","ποια","τι","πως","πώς","βρες",
    "ποιες","ποιους","νομος","νομοι","ν",
}


def _query_keywords(query: str, intent_keywords: list[str] | None) -> list[str]:
    kws: list[str] = []
    if intent_keywords:
        kws.extend(k for k in intent_keywords if isinstance(k, str) and k.strip())
    if not kws:
        for tok in re.split(r"[\s,.;:!?\"'()]+", query):
            n = _normalize(tok)
            if len(n) >= 3 and n not in _GREEK_STOPWORDS:
                kws.append(tok)
    seen = set()
    out: list[str] = []
    for k in kws:
        n = _normalize(k)
        if n and n not in seen:
            seen.add(n)
            out.append(k)
    return out


def _score_row(row: ListingRow, keywords: list[str], query_norm: str) -> float:
    if not keywords:
        return 0.0
    haystack = _normalize(
        " ".join(
            [
                row.get("description", ""),
                row.get("fek_title_raw", ""),
                row.get("kind", ""),
                str(row.get("number") or ""),
            ]
        )
    )
    score = 0.0
    for kw in keywords:
        n = _normalize(kw)
        if not n:
            continue
        score += haystack.count(n)
    num = row.get("number")
    if num is not None and re.search(rf"\b{num}\b", query_norm):
        score += 5.0
    return score


def _select_candidates(
    query: str,
    intent: dict,
    year: int | None,
    top_n: int,
) -> list[ListingRow]:
    rows = _load_listings()
    filtered = [r for r in rows if _row_passes_hard_filters(r, intent, year)]
    if not filtered:
        return []

    intent_keywords = intent.get("keywords") if isinstance(intent.get("keywords"), list) else None
    keywords = _query_keywords(query, intent_keywords)
    query_norm = _normalize(query)

    has_uniqueness_filter = any(
        intent.get(k) is not None
        for k in ("number", "fek_number")
    )
    if has_uniqueness_filter and len(filtered) <= top_n:
        return filtered

    scored = [(r, _score_row(r, keywords, query_norm)) for r in filtered]
    has_keywords = bool(keywords)
    has_constraints = year is not None or any(
        intent.get(k) is not None
        for k in (
            "date_from", "date_to", "year", "kind", "number",
            "fek_series", "fek_number", "fek_year", "min_pages", "max_pages",
        )
    )

    if has_keywords:
        matched = [(r, s) for r, s in scored if s > 0]
        if matched:
            scored = matched
        elif not has_constraints:
            return []
    elif not has_constraints:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_n]]


class ListingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ListingAgent")
        self.llm = ChatOpenAI(
            model=settings.metadata_llm_model,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}},
            api_key=settings.openai_api_key,
        )

    def _extract_intent(self, query: str) -> dict[str, Any]:
        today = date.today().isoformat()
        try:
            resp = self.llm.invoke(
                [
                    SystemMessage(content=_INTENT_SYSTEM),
                    HumanMessage(
                        content=_INTENT_USER_TEMPLATE.format(today=today, query=query)
                    ),
                ]
            )
            raw = resp.content or "{}"
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as e:
            _console.print(
                f"[yellow]ListingAgent intent extraction failed:[/yellow] {e}"
            )
            return {}

    def execute(
        self,
        state: dict[str, Any],
        top_candidates: int = 5,
        chunks_per_candidate: int = 3,
    ) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        embedding = state.get("query_embedding") or []
        year = state.get("year")
        k = state.get("top_k")

        if not embedding:
            return {"listing_results": []}

        intent = self._extract_intent(query)
        candidates = _select_candidates(query, intent, year, top_candidates)
        if not candidates:
            return {"listing_results": []}

        results: list[Document] = []
        seen_ids: set[Any] = set()
        for row in candidates:
            chunks = hybrid_search(
                query_text=query,
                query_embedding=embedding,
                match_count=chunks_per_candidate,
                filter={"source": row["pdf_basename"]},
            )
            for c in chunks:
                meta = c.get("metadata") or {}
                if c.get("id") is not None:
                    cid: tuple = ("id", c["id"])
                else:
                    cid = (
                        "sp",
                        meta.get("source"),
                        (meta.get("pages") or [None])[0],
                        (c.get("content") or "")[:64],
                    )
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                doc = dict_to_document(c)
                doc.metadata["_listing_match"] = {
                    "kind": row.get("kind"),
                    "number": row.get("number"),
                    "description": row.get("description"),
                    "fek_title": row.get("fek_title_raw"),
                    "fek_date": row["fek_date"].isoformat() if row.get("fek_date") else None,
                    "pages_count": row.get("pages_count"),
                }
                results.append(doc)

        if k is not None:
            results = results[:k]
        return {"listing_results": results}


@lru_cache(maxsize=1)
def get_listing_agent() -> ListingAgent:
    return ListingAgent()
