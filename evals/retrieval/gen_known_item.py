"""§2.2 — build known-item retrieval truth from the listing manifests.

A known-item query names a specific document ("Ν. 5263", "ΦΕΚ Α 238/2025"); the
ListingAgent should return chunks from exactly that ΦΕΚ. Ground truth is the
listing row's PDF basename. We sample rows whose PDF is actually INGESTED (checked
per-row, so we don't assert on documents the corpus doesn't have) in the target
years.

Usage (read-only; needs Supabase + local downloads/listing-items-*.md):
    python -m evals.retrieval.gen_known_item --n 15
Writes evals/datasets/known_item.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from evals._util import DATASETS_DIR
from src.config import settings
from src.rag.agents.listing_agent import _load_listings
from src.retrieval.store import _read_client


def _is_ingested(basename: str) -> bool:
    r = (_read_client().table(settings.supabase_table)
         .select("id", count="exact").eq("metadata->>source", basename).limit(1).execute())
    return (r.count or 0) > 0


def _row_year(r) -> int | None:
    if r.get("fek_year"):
        return r["fek_year"]
    if r.get("fek_date"):
        return r["fek_date"].year
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    ap.add_argument("--out", default="known_item.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [r for r in _load_listings()
            if r.get("number") and r.get("kind") and r.get("pdf_basename")
            and _row_year(r) in args.years]
    rng.shuffle(rows)
    print(f"{len(rows)} candidate listing rows in {args.years}; verifying ingestion...")

    tmp = (DATASETS_DIR / args.out).with_suffix(".jsonl.tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            if written >= args.n:
                break
            bn = r["pdf_basename"]
            if not _is_ingested(bn):
                continue
            f.write(json.dumps({
                "query": f"{r['kind']} {r['number']}",
                "expected_source": bn,
                "kind": r["kind"], "number": r["number"], "year": _row_year(r),
                "fek_title": r.get("fek_title_raw"),
            }, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no ingested listing rows found for the given years")
    os.replace(tmp, DATASETS_DIR / args.out)
    print(f"wrote {written} known-item queries -> {DATASETS_DIR / args.out}")


if __name__ == "__main__":
    main()
