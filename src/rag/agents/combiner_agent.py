from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.config import settings
from src.rag.agents.base_agent import BaseAgent


_console = Console()


_SYSTEM_PROMPT = (
    "Είσαι βοηθός για την ελληνική νομοθεσία. Απαντάς ΜΟΝΟ με βάση τα παρακάτω "
    "αποσπάσματα ΦΕΚ. Παρέθεσε αναφορές με τη μορφή [n] όπου n είναι ο αριθμός "
    "της πηγής. Αν δεν υπάρχει αρκετή πληροφορία στα αποσπάσματα, πες το ρητά "
    "αντί να μαντέψεις."
)


def _doc_key(d: Document) -> tuple:
    meta = d.metadata or {}
    if meta.get("id") is not None:
        return ("id", meta["id"])
    page = (meta.get("pages") or [None])[0]
    return ("sp", meta.get("source"), page, (d.page_content or "")[:64])


def _rank_score(d: Document) -> float:
    try:
        return float((d.metadata or {}).get("rank") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fuse(
    chunk_results: list[Document],
    listing_results: list[Document],
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
            merged_meta = dict(existing.metadata)
            merged_meta["_listing_match"] = d.metadata["_listing_match"]
            chosen[key] = Document(page_content=existing.page_content, metadata=merged_meta)

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
        header = f"[{i}] {filename}{page_str}"
        if title:
            header += f" — {title}"
        elif listing.get("description"):
            header += f" — {listing['description'][:120]}"
        if listing.get("fek_title"):
            header += f" (ΦΕΚ {listing['fek_title']})"
        lines.append(f"{header}\n{src.page_content}")
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
        query = state.get("rewritten_query") or state["query"]
        chunk_results = state.get("chunk_results") or []
        listing_results = state.get("listing_results") or []

        fused = _fuse(chunk_results, listing_results)
        if not fused:
            answer = "Δεν βρέθηκαν σχετικά αποσπάσματα στη βάση."
            return {
                "answer": answer,
                "sources": [],
                "messages": [AIMessage(content=answer)],
            }

        top = fused[: settings.top_k]
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

        return {
            "answer": answer_text,
            "sources": top,
            "messages": [AIMessage(content=answer_text)],
        }


@lru_cache(maxsize=1)
def get_combiner_agent() -> CombinerAgent:
    return CombinerAgent()
