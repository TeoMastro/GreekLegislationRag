import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.config import settings
from src.rag.agents.base_agent import BaseAgent


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _strip_invalid_citations(answer: str, num_sources: int) -> str:
    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if 1 <= n <= num_sources:
            return m.group(0)
        return ""

    cleaned = _CITATION_RE.sub(repl, answer)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)


_console = Console()


_SYSTEM_PROMPT = (
    "Είσαι βοηθός για την ελληνική νομοθεσία. Απαντάς ΜΟΝΟ με βάση τα παρακάτω "
    "αποσπάσματα ΦΕΚ. Παρέθεσε αναφορές με τη μορφή [n] όπου n είναι ο αριθμός "
    "της πηγής. Αν δεν υπάρχει αρκετή πληροφορία στα αποσπάσματα, πες το ρητά "
    "αντί να μαντέψεις. Όταν ένα απόσπασμα φέρει σημείωση «Σχέση γράφου», "
    "χρησιμοποίησέ την για να διατυπώσεις τη σχέση μεταξύ νόμων ρητά "
    "(π.χ. «ο Ν. X/Y τροποποιεί το άρθρο Z του Ν. A/B»)."
)


# Bonus added to rank score for graph-derived hits so they don't sink under
# semantic/full-text results that score on lexical similarity rather than
# the explicit relation the user asked about.
_RELATION_RANK_BOOST = 1.0


def _doc_key(d: Document) -> tuple:
    meta = d.metadata or {}
    if meta.get("id") is not None:
        return ("id", meta["id"])
    page = (meta.get("pages") or [None])[0]
    return ("sp", meta.get("source"), page, (d.page_content or "")[:64])


def _rank_score(d: Document) -> float:
    try:
        base = float((d.metadata or {}).get("rank") or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    if (d.metadata or {}).get("_relation_match"):
        # Relation matches are answering exactly the question type we boosted
        # for; weight them by classifier confidence so high-confidence edges
        # outrank lower-confidence ones.
        rel = d.metadata["_relation_match"]
        try:
            conf = float(rel.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        base += _RELATION_RANK_BOOST + conf
    return base


def _merge_meta(existing: Document, addition_key: str, addition: dict) -> Document:
    merged = dict(existing.metadata or {})
    merged[addition_key] = addition
    return Document(page_content=existing.page_content, metadata=merged)


def _fuse(
    chunk_results: list[Document],
    listing_results: list[Document],
    relation_results: list[Document],
) -> list[Document]:
    chosen: dict[tuple, Document] = {}

    for d in chunk_results:
        key = _doc_key(d)
        if key not in chosen or _rank_score(d) > _rank_score(chosen[key]):
            chosen[key] = d

    for d in listing_results:
        key = _doc_key(d)
        if key not in chosen:
            chosen[key] = d
            continue
        existing = chosen[key]
        if "_listing_match" in d.metadata and "_listing_match" not in existing.metadata:
            chosen[key] = _merge_meta(existing, "_listing_match", d.metadata["_listing_match"])

    for d in relation_results:
        key = _doc_key(d)
        if key not in chosen:
            chosen[key] = d
            continue
        existing = chosen[key]
        if "_relation_match" in d.metadata and "_relation_match" not in existing.metadata:
            chosen[key] = _merge_meta(existing, "_relation_match", d.metadata["_relation_match"])

    return sorted(chosen.values(), key=_rank_score, reverse=True)


def _format_context(sources: list[Document]) -> str:
    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        meta = src.metadata or {}
        filename = meta.get("source", "?")
        pages = meta.get("pages") or []
        page_str = f", σ. {pages[0]}" if pages else ""
        title = meta.get("title")
        listing = meta.get("_listing_match") or {}
        relation = meta.get("_relation_match") or {}
        header = f"[{i}] {filename}{page_str}"
        if title:
            header += f" — {title}"
        elif listing.get("description"):
            header += f" — {listing['description'][:120]}"
        if listing.get("fek_title"):
            header += f" (ΦΕΚ {listing['fek_title']})"
        body = src.page_content
        if relation:
            article = relation.get("target_article")
            article_str = f" άρθρο {article}" if article else ""
            note = (
                f"Σχέση γράφου: {relation.get('source_law')} "
                f"{relation.get('relation')} {relation.get('target_law')}{article_str} "
                f"(εμπιστοσύνη {float(relation.get('confidence') or 0.0):.2f})"
            )
            body = f"{note}\n{body}"
        lines.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(lines)


class CombinerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("CombinerAgent")
        self.llm = ChatOpenAI(
            model=settings.openai_chat_model,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("embedding_error"):
            answer = (
                "Η υπηρεσία ενσωματώσεων (embeddings) δεν είναι διαθέσιμη αυτή τη "
                "στιγμή, οπότε δεν ήταν δυνατή η ανάκτηση αποσπασμάτων. "
                "Δοκιμάστε ξανά σε λίγο."
            )
            return {
                "answer": answer,
                "sources": [],
                "messages": [AIMessage(content=answer)],
            }

        query = state.get("rewritten_query") or state["query"]
        chunk_results = state.get("chunk_results") or []
        listing_results = state.get("listing_results") or []
        relation_results = state.get("relation_results") or []

        fused = _fuse(chunk_results, listing_results, relation_results)
        if not fused:
            answer = "Δεν βρέθηκαν σχετικά αποσπάσματα στη βάση."
            return {
                "answer": answer,
                "sources": [],
                "messages": [AIMessage(content=answer)],
            }

        limit = state.get("top_k") or settings.top_k
        top = fused[:limit]
        context = _format_context(top)
        user_msg = f"Ερώτηση:\n{query}\n\nΑποσπάσματα ΦΕΚ:\n{context}"

        try:
            resp = self.llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            answer_text = resp.content or ""
        except Exception as e:
            _console.print(f"[red]CombinerAgent LLM call failed:[/red] {e}")
            raise

        answer_text = _strip_invalid_citations(answer_text, len(top))

        return {
            "answer": answer_text,
            "sources": top,
            "messages": [AIMessage(content=answer_text)],
        }


@lru_cache(maxsize=1)
def get_combiner_agent() -> CombinerAgent:
    return CombinerAgent()
