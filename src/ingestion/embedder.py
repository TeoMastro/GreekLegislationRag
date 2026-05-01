from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from src.config import settings


_MAX_TOKENS_PER_REQUEST = 250_000
_MAX_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(settings.openai_embedding_model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_request(batch: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(
        model=settings.openai_embedding_model,
        input=batch,
    )
    return [item.embedding for item in resp.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    enc = _encoding()
    out: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        n_tokens = len(enc.encode(text, disallowed_special=()))
        if batch and (
            len(batch) >= _MAX_BATCH_SIZE
            or batch_tokens + n_tokens > _MAX_TOKENS_PER_REQUEST
        ):
            out.extend(_embed_request(batch))
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += n_tokens
    if batch:
        out.extend(_embed_request(batch))
    return out
