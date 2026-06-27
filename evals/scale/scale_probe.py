"""§3 Scale probe — read-only signals that bite as the corpus grows toward ~1M.

Runs the synthetic gold set through the live index and reports:
  1. Query latency (p50/p95/max) for hybrid_search — the cost that compounds at
     scale and against a remote Supabase.
  2. Filtered vs unfiltered document recall — does `metadata` year-filtering
     degrade recall? pgvector HNSW POST-filters (find neighbours, then drop
     non-matches), so a filter can return fewer true neighbours at scale.
  3. Candidate-pool headroom — doc_recall@10 vs doc_recall@50. If @50 ≫ @10, the
     `least(match_count*2, 50)` candidate cap and/or HNSW `ef_search` are throttling
     recall; that gap widens as relevant chunks get rarer in a bigger corpus.
  4. RAM projection — static calc of vector memory at the current dim and at 1M
     chunks, against the 8 GB constraint.

Read-only (no writes); safe against the live DB. Exploratory tuning tool, not a
CI gate. Usage:
    python -m evals.scale.scale_probe --k 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rich.console import Console
from rich.table import Table

from evals._util import DATASETS_DIR, write_report
from src.config import settings
from src.ingestion.embedder import embed_texts
from src.retrieval.store import count_chunks, hybrid_search

_console = Console()


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[i]


def _doc_rank(results: list[dict], target_src: str) -> int | None:
    for i, d in enumerate(results):
        if (d.get("metadata") or {}).get("source") == target_src:
            return i + 1
    return None


def _timed_search(**kw) -> tuple[list[dict], float, bool]:
    """Return (results, latency_s, timed_out). A statement timeout / API error is
    captured (not raised) — a query that can't complete in the DB's time budget is
    itself a scale signal, not a reason to abort the probe."""
    t0 = time.perf_counter()
    try:
        return hybrid_search(**kw), time.perf_counter() - t0, False
    except Exception:  # noqa: BLE001 — postgrest APIError (e.g. 57014 timeout)
        return [], time.perf_counter() - t0, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=50, help="match_count (also the headroom ceiling)")
    ap.add_argument("--filtered-sample", type=int, default=10,
                    help="run the year-filtered leg on only the first N queries "
                         "(it can hit the DB statement timeout, ~60s each)")
    ap.add_argument("--out", default="scale_probe.json")
    args = ap.parse_args()

    gold = [json.loads(l) for l in (DATASETS_DIR / "synthetic_queries.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    if not gold:
        raise SystemExit("empty gold set — run synthetic_gen.py first")

    n_chunks = count_chunks()
    _console.print(f"[bold]Scale probe[/bold] · {len(gold)} queries · k={args.k} · corpus={n_chunks} chunks")

    lat_unf, lat_flt = [], []
    hit10 = hit50 = hit10_flt = 0
    to_unf = to_flt = 0  # timeout counts
    evaluated = flt_evaluated = 0

    for r in gold:
        emb = embed_texts([r["question"]])
        if not emb:
            continue
        evaluated += 1
        src = r.get("source")

        res, lat, timed_out = _timed_search(
            query_text=r["question"], query_embedding=emb[0], match_count=args.k)
        lat_unf.append(lat)
        to_unf += int(timed_out)
        dr = _doc_rank(res, src)
        if dr and dr <= 10:
            hit10 += 1
        if dr and dr <= args.k:
            hit50 += 1

        # Filtered leg only on a bounded sample — it can block on the statement
        # timeout, so running all 50 could take ~50 min.
        if flt_evaluated < args.filtered_sample:
            yr = r.get("year")
            res_f, lat_f, timed_out_f = _timed_search(
                query_text=r["question"], query_embedding=emb[0], match_count=args.k,
                filter={"year": yr} if yr is not None else {})
            lat_flt.append(lat_f)
            to_flt += int(timed_out_f)
            drf = _doc_rank(res_f, src)
            if drf and drf <= 10:
                hit10_flt += 1
            flt_evaluated += 1

    n = evaluated or 1
    # RAM projection: float32 vectors only (HNSW graph adds ~1.5-3x on top).
    def gb(dim: int, rows: int) -> float:
        return rows * dim * 4 / 1e9

    report = {
        "n": evaluated,
        "corpus_chunks": n_chunks,
        "k": args.k,
        "latency_unfiltered_s": {"p50": round(_pct(lat_unf, 50), 3), "p95": round(_pct(lat_unf, 95), 3),
                                 "max": round(max(lat_unf), 3) if lat_unf else 0},
        "latency_filtered_s": {"p50": round(_pct(lat_flt, 50), 3), "p95": round(_pct(lat_flt, 95), 3),
                               "max": round(max(lat_flt), 3) if lat_flt else 0},
        "doc_recall@10": round(hit10 / n, 4),
        f"doc_recall@{args.k}": round(hit50 / n, 4),
        "headroom@10_to@k": round((hit50 - hit10) / n, 4),
        "filtered_sample": flt_evaluated,
        "doc_recall@10_filtered": round(hit10_flt / flt_evaluated, 4) if flt_evaluated else None,
        "timeouts_unfiltered": to_unf,
        "timeouts_filtered": to_flt,
        "ram_projection_gb_vectors_only": {
            "current_dim_current_corpus": round(gb(settings.openai_embedding_dimension, n_chunks), 2),
            "1536d_at_1M": round(gb(1536, 1_000_000), 2),
            "3072d_at_1M": round(gb(3072, 1_000_000), 2),
        },
    }
    write_report(args.out, report)

    t = Table(title="Scale probe")
    t.add_column("metric"); t.add_column("value")
    for k, v in report.items():
        t.add_row(k, json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v))
    _console.print(t)
    _console.print(
        "[dim]Note: RAM figures are raw float32 vectors only; the HNSW graph adds "
        "~1.5-3x on top, and 8 GB total must also hold Postgres + OS.[/dim]"
    )


if __name__ == "__main__":
    main()
