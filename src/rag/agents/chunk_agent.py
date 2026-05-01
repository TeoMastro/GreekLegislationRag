from functools import lru_cache
from typing import Any

from src.rag.agents.base_agent import BaseAgent
from src.rag.retriever import SupabaseHybridRetriever


class ChunkAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ChunkAgent")
        self.retriever = SupabaseHybridRetriever()

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        embedding = state.get("query_embedding") or []
        if not embedding:
            return {"chunk_results": []}

        year = state.get("year")
        k = state.get("top_k")
        filter_ = {"year": year} if year is not None else {}
        docs = self.retriever.retrieve_with_embedding(
            query=query,
            query_embedding=embedding,
            k=k,
            filter=filter_,
        )
        return {"chunk_results": docs}


@lru_cache(maxsize=1)
def get_chunk_agent() -> ChunkAgent:
    return ChunkAgent()
