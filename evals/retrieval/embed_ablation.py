"""§2.4 Embedding-model A/B — does a stronger model fix the recall gap?

The corpus is stored as text-embedding-3-small vectors, so you cannot compare a
different model by only re-embedding the query (different vector spaces). This
script instead builds an in-memory MINI-INDEX — the gold targets plus a sample of
distractor chunks — and re-embeds BOTH chunks and queries with each candidate
model, then measures semantic-only document recall.

Caveats (read before trusting numbers):
  - The mini-index (~1-2k chunks) is far smaller than the 57k production index, so
    ABSOLUTE recall here is inflated and NOT comparable to ablation.py. Only the
    RELATIVE gap between models on the SAME mini-index is meaningful.
  - Semantic-only (no FTS leg). It answers "is model B's embedding better on Greek
    legislative text than model A?", which is the lever for the genuine recall miss.

Usage:
    python -m evals.retrieval.embed_ablation --distractors 1500 \
        --models text-embedding-3-small text-embedding-3-large
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from evals._util import DATASETS_DIR, write_report
from src.config import settings
from src.retrieval.store import fetch_chunks_by_ids, iter_chunks

_console = Console()
_MIN_CHARS = 200
_EMBED_BATCH = 128


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _embed(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = [t[:8000] for t in texts[i : i + _EMBED_BATCH]]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
        _console.print(f"    embedded {min(i + _EMBED_BATCH, len(texts))}/{len(texts)} ({model})")
    arr = np.array(out, dtype=np.float32)
    # L2-normalize so a dot product == cosine similarity.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _build_mini_index(gold: list[dict], n_distractors: int, seed: int) -> list[dict]:
    """Gold target chunks + a random distractor sample (deduped by id)."""
    rng = random.Random(seed)
    target_ids = [int(r["chunk_id"]) for r in gold]
    targets = fetch_chunks_by_ids(target_ids)
    have = {int(t["id"]) for t in targets}

    pool: list[dict] = []
    for row in iter_chunks():
        if int(row["id"]) in have:
            continue
        if len((row.get("content") or "").strip()) < _MIN_CHARS:
            continue
        pool.append(row)
    rng.shuffle(pool)
    return targets + pool[:n_distractors]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distractors", type=int, default=1500)
    ap.add_argument("--models", nargs="+",
                    default=["text-embedding-3-small", "text-embedding-3-large"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="embed_ablation.json")
    args = ap.parse_args()

    gold = [json.loads(l) for l in (DATASETS_DIR / "synthetic_queries.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    client = _client()

    _console.print(f"building mini-index: {len(gold)} targets + {args.distractors} distractors")
    corpus = _build_mini_index(gold, args.distractors, args.seed)
    corpus_ids = [int(c["id"]) for c in corpus]
    corpus_src = [(c.get("metadata") or {}).get("source") for c in corpus]
    corpus_texts = [c.get("content") or "" for c in corpus]
    q_texts = [r["question"] for r in gold]
    _console.print(f"mini-index size: {len(corpus)} chunks")

    results: dict[str, dict] = {}
    for model in args.models:
        _console.print(f"[bold]model: {model}[/bold]")
        doc_emb = _embed(client, corpus_texts, model)          # (C, d)
        q_emb = _embed(client, q_texts, model)                 # (Q, d)
        sims = q_emb @ doc_emb.T                                # (Q, C) cosine

        doc_hits = {1: 0, 5: 0, 10: 0}
        exact_hits10 = 0
        rr_sum = 0.0
        for qi, r in enumerate(gold):
            order = np.argsort(-sims[qi])                       # best first
            ranked_src = [corpus_src[j] for j in order]
            ranked_ids = [corpus_ids[j] for j in order]
            # document-level
            dr = next((i + 1 for i, s in enumerate(ranked_src) if s == r.get("source")), None)
            if dr:
                for k in doc_hits:
                    if dr <= k:
                        doc_hits[k] += 1
            # exact chunk
            tgt = int(r["chunk_id"])
            er = next((i + 1 for i, cid in enumerate(ranked_ids) if cid == tgt), None)
            if er:
                if er <= 10:
                    exact_hits10 += 1
                rr_sum += 1.0 / er

        n = len(gold)
        results[model] = {
            "doc_recall@1": round(doc_hits[1] / n, 4),
            "doc_recall@5": round(doc_hits[5] / n, 4),
            "doc_recall@10": round(doc_hits[10] / n, 4),
            "exact_recall@10": round(exact_hits10 / n, 4),
            "exact_mrr": round(rr_sum / n, 4),
        }

    write_report(args.out, {"mini_index_size": len(corpus), "n_queries": len(gold), "models": results})

    table = Table(title=f"Embedding A/B (mini-index={len(corpus)}, semantic-only, n={len(gold)})")
    for col in ["model", "doc@1", "doc@5", "doc@10", "exact@10", "mrr"]:
        table.add_column(col)
    for model, m in results.items():
        table.add_row(model, str(m["doc_recall@1"]), str(m["doc_recall@5"]),
                      str(m["doc_recall@10"]), str(m["exact_recall@10"]), str(m["exact_mrr"]))
    _console.print(table)


if __name__ == "__main__":
    main()
