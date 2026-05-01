from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    query: str
    rewritten_query: str
    query_embedding: list[float]
    year: int | None
    top_k: int | None

    chunk_results: list[Document]
    listing_results: list[Document]

    answer: str
    sources: list[Document]

    embedding_error: str | None

    messages: Annotated[list[BaseMessage], add_messages]


def create_initial_state(
    query: str,
    year: int | None = None,
    top_k: int | None = None,
) -> RAGState:
    return RAGState(
        query=query,
        rewritten_query="",
        query_embedding=[],
        year=year,
        top_k=top_k,
        chunk_results=[],
        listing_results=[],
        answer="",
        sources=[],
        embedding_error=None,
        messages=[],
    )
