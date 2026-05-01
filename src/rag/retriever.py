from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from src.config import settings
from src.ingestion.embedder import embed_texts
from src.retrieval.store import hybrid_search


def dict_to_document(d: dict[str, Any]) -> Document:
    meta = dict(d.get("metadata") or {})
    if "id" in d and "id" not in meta:
        meta["id"] = d["id"]
    if "similarity" in d:
        meta["similarity"] = d["similarity"]
    if "rank" in d:
        meta["rank"] = d["rank"]
    return Document(page_content=d.get("content", "") or "", metadata=meta)


class SupabaseHybridRetriever(BaseRetriever):
    """LangChain retriever wrapping the project's match_documents_hybrid RPC."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int | None = None
    filter: dict[str, Any] | None = None

    def retrieve_with_embedding(
        self,
        query: str,
        query_embedding: list[float],
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        if filter is None:
            filter = self.filter or {}
        rows = hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            match_count=k or self.k or settings.top_k,
            filter=filter,
        )
        return [dict_to_document(r) for r in rows]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embeddings = embed_texts([query])
        if not embeddings:
            return []
        return self.retrieve_with_embedding(
            query, embeddings[0], k=self.k, filter=self.filter
        )
