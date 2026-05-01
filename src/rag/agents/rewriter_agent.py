from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.config import settings
from src.ingestion.embedder import embed_texts
from src.rag.agents.base_agent import BaseAgent


_console = Console()


_REWRITER_SYSTEM = (
    "Είσαι βοηθός που ξαναγράφει follow-up ερωτήσεις σε αυτοτελή μορφή, βάσει του "
    "ιστορικού της συνομιλίας. Επιστρέφεις ΜΟΝΟ την ξαναγραμμένη ερώτηση, χωρίς "
    "επεξήγηση. Αν η ερώτηση είναι ήδη αυτοτελής, επίστρεψέ την ως έχει."
)


def _format_history(messages: list, max_turns: int = 4) -> str:
    recent = messages[-max_turns * 2 :] if max_turns > 0 else messages
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"Χρήστης: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Βοηθός: {m.content}")
    return "\n".join(lines)


class RewriterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("RewriterAgent")
        self.llm = ChatOpenAI(
            model=settings.rewriter_llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        messages = state.get("messages") or []
        prior = messages[:-1] if messages else []

        if not settings.enable_query_rewriter or not prior:
            rewritten = query
        else:
            history = _format_history(prior)
            try:
                resp = self.llm.invoke(
                    [
                        SystemMessage(content=_REWRITER_SYSTEM),
                        HumanMessage(
                            content=(
                                f"Ιστορικό συνομιλίας:\n{history}\n\n"
                                f"Νέα ερώτηση: {query}\n\n"
                                f"Ξαναγραμμένη αυτοτελής ερώτηση:"
                            )
                        ),
                    ]
                )
                rewritten = (resp.content or query).strip() or query
            except Exception as e:
                _console.print(
                    f"[yellow]RewriterAgent failed, using original query:[/yellow] {e}"
                )
                rewritten = query

        try:
            embeddings = embed_texts([rewritten])
        except Exception as e:
            _console.print(
                f"[red]RewriterAgent embedding call failed:[/red] {e}"
            )
            return {
                "rewritten_query": rewritten,
                "query_embedding": [],
                "embedding_error": str(e) or type(e).__name__,
            }

        embedding = embeddings[0] if embeddings else []
        if not embedding:
            return {
                "rewritten_query": rewritten,
                "query_embedding": [],
                "embedding_error": "embedding service returned empty vector",
            }

        return {
            "rewritten_query": rewritten,
            "query_embedding": embedding,
            "embedding_error": None,
        }


@lru_cache(maxsize=1)
def get_rewriter_agent() -> RewriterAgent:
    return RewriterAgent()
