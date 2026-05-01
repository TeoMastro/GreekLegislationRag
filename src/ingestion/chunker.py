from functools import lru_cache
from typing import Iterator

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument
import tiktoken

from src.config import settings


@lru_cache(maxsize=1)
def _chunker() -> HybridChunker:
    try:
        encoding = tiktoken.encoding_for_model(settings.openai_embedding_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = OpenAITokenizer(tokenizer=encoding, max_tokens=settings.chunk_tokens)
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def chunk_document(doc: DoclingDocument) -> Iterator[dict]:
    chunker = _chunker()
    for chunk in chunker.chunk(doc):
        text = chunker.contextualize(chunk=chunk)
        meta = chunk.meta.export_json_dict() if chunk.meta else {}
        headings = meta.get("headings") or []
        pages: list[int] = []
        for item in meta.get("doc_items", []) or []:
            for prov in item.get("prov", []) or []:
                page = prov.get("page_no")
                if page is not None and page not in pages:
                    pages.append(page)
        yield {
            "text": text,
            "headings": headings,
            "pages": pages,
        }
