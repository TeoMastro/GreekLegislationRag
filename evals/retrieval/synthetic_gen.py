"""§2.1 Synthetic query generator — builds a known-positive retrieval gold set.

For a stratified sample of chunks already in Supabase, ask the LLM to write a
Greek question answerable ONLY from that chunk. The source chunk is then the
known positive for that question, which lets `test_retrieval_recall.py` measure
recall@k / MRR without any human labelling.

CAVEAT (best practice): synthetic queries are systematically easier than real
user questions, so treat the resulting numbers as a *relative regression metric*
across config/model/embedding changes — not an absolute quality score. Fold in a
small set of real user queries as soon as you have them.

Usage (needs live OpenAI + Supabase + a populated corpus):
    python -m evals.retrieval.synthetic_gen --n 200 --per-year 40 --seed 7

Writes evals/datasets/synthetic_queries.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from openai import OpenAI

from evals._util import DATASETS_DIR
from src.config import settings
from src.retrieval.store import iter_chunks

_GEN_SYSTEM = (
    "Είσαι ειδικός στην ελληνική νομοθεσία. Διαβάζεις ΕΝΑ απόσπασμα ΦΕΚ και "
    "γράφεις ΜΙΑ φυσική ερώτηση στα ελληνικά, την οποία ΜΟΝΟ αυτό το απόσπασμα "
    "μπορεί να απαντήσει.\n"
    "ΚΑΝΟΝΕΣ:\n"
    "1. Η ερώτηση πρέπει να είναι ΑΥΤΟΤΕΛΗΣ — κατανοητή χωρίς να βλέπει κανείς το "
    "κείμενο. ΜΗΝ αναφέρεις «το απόσπασμα», «το παρόν», «στο κείμενο» κ.λπ.\n"
    "2. ΜΗΝ είναι απλή αναζήτηση τιμής σε πίνακα (π.χ. «ποια τιμή αντιστοιχεί στον "
    "κωδικό 1») — τέτοιες ερωτήσεις δεν ταυτοποιούν έγγραφο.\n"
    "3. Πρέπει να αφορά ΟΥΣΙΑΣΤΙΚΟ νομικό/κανονιστικό περιεχόμενο και να περιέχει "
    "αρκετό πλαίσιο (θέμα, φορέα, αντικείμενο) ώστε να εντοπίζεται το σωστό ΦΕΚ.\n"
    "4. Μην αντιγράφεις αυτούσιες μεγάλες φράσεις (να μην είναι λεκτικό ταίριασμα).\n"
    "Επιστρέφεις ΜΟΝΟ την ερώτηση, χωρίς εισαγωγικά ή επεξήγηση."
)

_MIN_CHARS = 200  # skip tiny/boilerplate chunks that can't anchor a unique question


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _sample_chunks(
    n: int, per_year: int, per_source: int, seed: int, max_scan: int | None = None
) -> list[dict]:
    """Stratified sample: bucket by year, then take up to `per_year` chunks per
    year while capping `per_source` chunks per PDF.

    The per-source cap is the key fix over a naive sample: legislative corpora
    have a few very large, table-heavy documents that would otherwise dominate the
    gold set and skew recall (see the 2023 table-lookup pathology). Capping per
    source forces document diversity.

    `max_scan` caps how many chunks are pulled from Supabase before sampling —
    use it for cheap smoke runs (biased toward earlier rows). Leave it None for a
    representative gold set.
    """
    rng = random.Random(seed)
    by_year: dict[str, list[dict]] = defaultdict(list)
    scanned = 0
    for row in iter_chunks():
        scanned += 1
        content = (row.get("content") or "").strip()
        if len(content) < _MIN_CHARS:
            continue
        year = str((row.get("metadata") or {}).get("year") or "?")
        by_year[year].append(row)
        if max_scan is not None and scanned >= max_scan:
            break

    picked: list[dict] = []
    for year, rows in by_year.items():
        rng.shuffle(rows)
        src_count: dict[str, int] = defaultdict(int)
        taken = 0
        for row in rows:
            src = (row.get("metadata") or {}).get("source")
            if src_count[src] >= per_source:
                continue
            src_count[src] += 1
            picked.append(row)
            taken += 1
            if taken >= per_year:
                break
    rng.shuffle(picked)
    return picked[:n]


def _make_question(client: OpenAI, content: str) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=settings.metadata_llm_model,
            messages=[
                {"role": "system", "content": _GEN_SYSTEM},
                {"role": "user", "content": content[:6000]},
            ],
            temperature=0.2,
        )
        q = (resp.choices[0].message.content or "").strip()
        return q or None
    except Exception as e:  # noqa: BLE001 — best-effort generation, skip failures
        print(f"  ! generation failed: {e}")
        return None


_VALIDATE_SYSTEM = (
    "Είσαι αυστηρός ελεγκτής ποιότητας ερωτήσεων αξιολόγησης. Σου δίνεται ένα "
    "ΑΠΟΣΠΑΣΜΑ ΦΕΚ και μια υποψήφια ΕΡΩΤΗΣΗ. Δέξου την ΜΟΝΟ αν πληροί ΟΛΑ:\n"
    "1. Είναι φυσική, καλοδιατυπωμένη ελληνική — όχι συντακτικά στρεβλή ή ασαφής.\n"
    "2. Είναι αυτοτελής — κατανοητή χωρίς να βλέπει κανείς το απόσπασμα (καμία "
    "αναφορά «στο απόσπασμα/κείμενο»).\n"
    "3. Απαντάται συγκεκριμένα από ΑΥΤΟ το απόσπασμα.\n"
    "4. Δεν είναι τετριμμένη αναζήτηση τιμής σε πίνακα ούτε υπερβολικά γενική.\n"
    "Επιστρέφεις ΜΟΝΟ JSON: {\"ok\": true|false}."
)


def _validate_question(client: OpenAI, question: str, content: str) -> bool:
    """Second-pass quality gate: reject garbled / non-self-contained / unanswerable
    questions so the gold set isn't polluted by bad inputs (which otherwise show up
    as spurious faithfulness/recall failures)."""
    try:
        resp = client.chat.completions.create(
            model=settings.metadata_llm_model,
            messages=[
                {"role": "system", "content": _VALIDATE_SYSTEM},
                {"role": "user", "content": f"ΑΠΟΣΠΑΣΜΑ:\n{content[:6000]}\n\nΕΡΩΤΗΣΗ:\n{question}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return bool(json.loads(resp.choices[0].message.content or "{}").get("ok"))
    except Exception as e:  # noqa: BLE001 — on judge error, keep the question
        print(f"  ! validation failed (keeping): {e}")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="total questions to generate")
    ap.add_argument("--per-year", type=int, default=40, help="cap of chunks per year")
    ap.add_argument("--per-source", type=int, default=2, help="cap of chunks per PDF (diversity)")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed (reproducible)")
    ap.add_argument("--oversample", type=int, default=3,
                    help="sample N*oversample candidates; the quality gate discards bad ones")
    ap.add_argument("--no-validate", action="store_true", help="skip the quality gate")
    ap.add_argument("--max-scan", type=int, default=None,
                    help="cap chunks pulled before sampling (cheap smoke runs)")
    ap.add_argument("--out", default="synthetic_queries.jsonl")
    args = ap.parse_args()

    client = _client()
    # Oversample candidates so the quality gate can discard garbled/unanswerable
    # questions and still reach --n. per_year is set high so the year cap doesn't bind.
    target = args.n * args.oversample
    chunks = _sample_chunks(target, target, args.per_source, args.seed, args.max_scan)
    print(f"sampled {len(chunks)} candidate chunks; generating + validating to reach {args.n}...")

    out_path = DATASETS_DIR / args.out
    # Write to a temp file and atomically replace at the end, so an existing gold
    # set is NEVER destroyed by a slow/failed/empty run (the file is untracked).
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    written = rejected = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in chunks:
            if written >= args.n:
                break
            content = row.get("content") or ""
            q = _make_question(client, content)
            if not q:
                continue
            if not args.no_validate and not _validate_question(client, q, content):
                rejected += 1
                continue
            meta = row.get("metadata") or {}
            f.write(
                json.dumps(
                    {
                        "chunk_id": int(row["id"]),
                        "source": meta.get("source"),
                        "year": meta.get("year"),
                        "question": q,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            f.flush()  # so partial progress is visible/recoverable mid-run

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        print("wrote 0 rows; left existing gold set untouched")
        return
    os.replace(tmp_path, out_path)
    print(f"wrote {written} rows -> {out_path}  (rejected {rejected} by quality gate)")


if __name__ == "__main__":
    main()
