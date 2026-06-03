"""§2.3 — build ground truth for the relational-graph eval from the live graph.

Samples real focal laws (kind Ν.) that have outgoing citation edges, and records
the exact target set the RelationAgent SHOULD return for an outgoing, all-relations
query — i.e. `citations_from_law(key, relations=None, min_confidence=0.5,
limit=relation_top_k)`. Generating truth from the same RPC the agent uses means
edge-recall < 1.0 can only come from a routing/hydration failure, not a truth
mismatch — which is exactly what the eval is meant to catch.

Usage (read-only; needs Supabase):
    python -m evals.graph.gen_relation_truth --n 15
Writes evals/datasets/relation_graph_truth.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from evals._util import DATASETS_DIR
from src.config import settings
from src.retrieval.store import _read_client, citations_from_law


def _all_nodes() -> dict[int, dict]:
    client = _read_client()
    out: dict[int, dict] = {}
    offset, page = 0, 1000
    while True:
        rows = (client.table("law_nodes")
                .select("id,canonical_key,kind,number,year")
                .range(offset, offset + page - 1).execute().data or [])
        if not rows:
            break
        for r in rows:
            out[int(r["id"])] = r
        if len(rows) < page:
            break
        offset += len(rows)
    return out


def _source_ids_with_edges() -> list[int]:
    client = _read_client()
    seen: set[int] = set()
    offset, page = 0, 1000
    while True:
        rows = (client.table("law_citations")
                .select("source_law_id,confidence")
                .gte("confidence", settings.citation_min_confidence)
                .range(offset, offset + page - 1).execute().data or [])
        if not rows:
            break
        for r in rows:
            if r.get("source_law_id") is not None:
                seen.add(int(r["source_law_id"]))
        if len(rows) < page:
            break
        offset += len(rows)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15, help="focal laws to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-targets", type=int, default=2)
    ap.add_argument("--max-targets", type=int, default=20,
                    help="skip laws with more edges than relation_top_k to avoid the cap")
    ap.add_argument("--out", default="relation_graph_truth.jsonl")
    args = ap.parse_args()

    nodes = _all_nodes()
    rng = random.Random(args.seed)
    src_ids = _source_ids_with_edges()
    # Prefer Ν. laws (cleanest natural phrasing).
    candidates = [nodes[i] for i in src_ids if i in nodes and nodes[i].get("kind") == "Ν."]
    rng.shuffle(candidates)
    print(f"{len(candidates)} candidate Ν. focal laws with edges; building truth...")

    tmp = (DATASETS_DIR / args.out).with_suffix(".jsonl.tmp")
    written = 0
    with tmp.open("w", encoding="utf-8") as f:
        for node in candidates:
            if written >= args.n:
                break
            key = node["canonical_key"]
            edges = citations_from_law(key, None, settings.citation_min_confidence,
                                       settings.relation_top_k)
            targets = sorted({e["target_key"] for e in edges if e.get("target_key")})
            if not (args.min_targets <= len(targets) <= args.max_targets):
                continue
            number, year = node["number"], node["year"]
            f.write(json.dumps({
                "focal_key": key,
                "kind": node["kind"], "number": number, "year": year,
                "query": f"Ποιους νόμους επηρεάζει ο ν. {number}/{year};",
                "targets": targets,
            }, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no suitable focal laws found")
    os.replace(tmp, DATASETS_DIR / args.out)
    print(f"wrote {written} focal laws -> {DATASETS_DIR / args.out}")


if __name__ == "__main__":
    main()
