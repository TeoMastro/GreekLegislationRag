"""§2.4 Retrieval ablation — justify (or overturn) the hybrid-search config.

Runs the synthetic gold set through several retrieval configurations and reports
document-level recall for each, so the production defaults
(full_text_weight=1, semantic_weight=1, rrf_k=50) rest on evidence rather than a
guess. This is an exploratory TUNING tool (emits a comparison report), not a
pass/fail CI gate.

What it isolates:
  - semantic-only vs full-text-only vs hybrid (each leg's contribution)
  - semantic/FTS weight ratio
  - rrf_k (the rank-fusion smoothing constant)

Query embeddings are computed once and reused across configs (only the SQL
fusion params change), so cost is ~one embed per query + one RPC per
(query × config).

Usage (needs live OpenAI + Supabase + the gold set):
    python -m evals.retrieval.ablation            # default config grid
    python -m evals.retrieval.ablation --k 10
"""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from evals._util import DATASETS_DIR, write_report
from src.ingestion.embedder import embed_texts
from src.retrieval.store import hybrid_search

_console = Console()

# (name, full_text_weight, semantic_weight, rrf_k). The default production config
# is flagged so the table makes the comparison obvious.
CONFIGS = [
    ("semantic_only", 0.0, 1.0, 50),
    ("fts_only", 1.0, 0.0, 50),
    ("hybrid_balanced*", 1.0, 1.0, 50),   # * = current production default
    ("hybrid_semantic_2x", 1.0, 2.0, 50),
    ("hybrid_fts_2x", 2.0, 1.0, 50),
    ("hybrid_rrf_k20", 1.0, 1.0, 20),
    ("hybrid_rrf_k100", 1.0, 1.0, 100),
]


def _load_gold() -> list[dict]:
    path = DATASETS_DIR / "synthetic_queries.jsonl"
    if not path.exists():
        raise SystemExit("run synthetic_gen.py first to build the gold set")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _doc_rank(results: list[dict], target_src: str) -> int | None:
    for i, d in enumerate(results):
        if (d.get("metadata") or {}).get("source") == target_src:
            return i + 1
    return None


def _exact_rank(results: list[dict], target_id: int) -> int | None:
    for i, d in enumerate(results):
        if d.get("id") is not None and int(d["id"]) == target_id:
            return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10, help="match_count (top-k)")
    ap.add_argument("--out", default="retrieval_ablation.json")
    args = ap.parse_args()

    gold = _load_gold()
    _console.print(f"[bold]Ablation[/bold] over {len(gold)} queries × {len(CONFIGS)} configs (k={args.k})")

    # Embed every query once, reuse across configs.
    embeddings: list[list[float]] = []
    for r in gold:
        emb = embed_texts([r["question"]])
        embeddings.append(emb[0] if emb else [])

    results_by_config: dict[str, dict] = {}
    for name, ftw, stw, rrf_k in CONFIGS:
        doc_hits = {1: 0, 5: 0, 10: 0}
        exact_hits10 = 0
        rr_sum = 0.0
        n = 0
        for r, emb in zip(gold, embeddings):
            if not emb:
                continue
            n += 1
            res = hybrid_search(
                query_text=r["question"],
                query_embedding=emb,
                match_count=args.k,
                full_text_weight=ftw,
                semantic_weight=stw,
                rrf_k=rrf_k,
            )
            dr = _doc_rank(res, r.get("source"))
            if dr:
                for k in doc_hits:
                    if dr <= k:
                        doc_hits[k] += 1
            er = _exact_rank(res, int(r["chunk_id"]))
            if er:
                if er <= 10:
                    exact_hits10 += 1
                rr_sum += 1.0 / er
        n = n or 1
        results_by_config[name] = {
            "full_text_weight": ftw,
            "semantic_weight": stw,
            "rrf_k": rrf_k,
            "doc_recall@1": round(doc_hits[1] / n, 4),
            "doc_recall@5": round(doc_hits[5] / n, 4),
            "doc_recall@10": round(doc_hits[10] / n, 4),
            "exact_recall@10": round(exact_hits10 / n, 4),
            "exact_mrr": round(rr_sum / n, 4),
            "n": n,
        }
        _console.print(f"  ran {name}: doc_recall@10={results_by_config[name]['doc_recall@10']}")

    write_report(args.out, {"k": args.k, "configs": results_by_config})

    # Comparison table, best doc_recall@10 first.
    table = Table(title=f"Retrieval ablation (k={args.k}, n={len(gold)} queries)")
    for col in ["config", "ftw", "stw", "rrf_k", "doc@1", "doc@5", "doc@10", "exact@10", "mrr"]:
        table.add_column(col)
    ordered = sorted(results_by_config.items(), key=lambda kv: kv[1]["doc_recall@10"], reverse=True)
    best = ordered[0][1]["doc_recall@10"]
    for name, m in ordered:
        style = "bold green" if m["doc_recall@10"] == best else None
        table.add_row(
            name, str(m["full_text_weight"]), str(m["semantic_weight"]), str(m["rrf_k"]),
            str(m["doc_recall@1"]), str(m["doc_recall@5"]), str(m["doc_recall@10"]),
            str(m["exact_recall@10"]), str(m["exact_mrr"]), style=style,
        )
    _console.print(table)


if __name__ == "__main__":
    main()
