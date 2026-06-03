"""Shared helpers for the eval suites: dataset loading + report emission.

Deterministic component evals (evals/component) import only `load_jsonl` and
`write_report`; neither touches the network, so the PR tier stays offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# evals/  (datasets live under evals/datasets, reports under evals/report)
EVALS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVALS_DIR / "datasets"
REPORT_DIR = EVALS_DIR / "report"


def load_jsonl(name: str) -> list[dict[str, Any]]:
    """Load a dataset file from evals/datasets. Blank lines and lines starting
    with `//` (comments) are skipped so datasets can stay self-documenting."""
    path = DATASETS_DIR / name
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"{name}:{i}: invalid JSON: {e}") from e
    return rows


def iter_jsonl(name: str) -> Iterator[dict[str, Any]]:
    yield from load_jsonl(name)


def write_report(name: str, payload: dict[str, Any]) -> Path:
    """Persist a metrics report as JSON under evals/report/ for trend tracking.

    No timestamp is added here (the harness has no clock authority and we want
    byte-stable output for diffing); CI is expected to stamp/commit these.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
