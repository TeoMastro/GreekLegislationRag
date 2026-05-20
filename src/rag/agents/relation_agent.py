"""RelationAgent — answers relational questions by querying the citation graph.

Triggers only when the user's question is structurally about *connections
between laws* (e.g. "ποιους νόμους τροποποιεί ο Ν. 5167/2024;", "ποιοι νόμοι
κατήργησαν τον ν. 4001/2011;"). For content questions the agent emits an
empty result, leaving the existing ChunkAgent + ListingAgent pipeline
unchanged.

When relational intent is detected, the agent:
  1. Resolves the focal law to a canonical_key.
  2. Calls citations_from_law / citations_to_law RPCs.
  3. Hydrates each citation row into a Document by fetching its source chunk
     (so the CombinerAgent can cite real ΦΕΚ text, not just a snippet).
  4. Attaches a _relation_match metadata block so the combiner's context
     formatter can phrase the answer in graph terms.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from src.citations.normalize import (
    CANONICAL_KINDS,
    canonical_key,
    canonicalize_kind,
)
from src.citations.extractor import RELATIONS
from src.config import settings
from src.rag.agents.base_agent import BaseAgent
from src.rag.retriever import dict_to_document
from src.retrieval.store import (
    citations_from_law,
    citations_to_law,
    fetch_chunks_by_ids,
)


_console = Console()


_INTENT_SYSTEM = (
    "Είσαι βοηθός που αναγνωρίζει ΑΝ μια ερώτηση πάνω στην ελληνική νομοθεσία "
    "αφορά ΣΧΕΣΕΙΣ μεταξύ νόμων (π.χ. ποιος νόμος τροποποιεί/καταργεί/αναφέρεται "
    "σε άλλον). Επιστρέφεις ΜΟΝΟ JSON."
)


_INTENT_USER_TEMPLATE = """Ερώτηση: {query}

Επέστρεψε JSON με τα κλειδιά:
- is_relational: true αν η ερώτηση αφορά σχέσεις μεταξύ νόμων, αλλιώς false
- direction: "outgoing" αν ρωτά τι κάνει ένας νόμος προς άλλους
              "incoming"  αν ρωτά ποιοι άλλοι επηρέασαν/αναφέρθηκαν σε έναν νόμο
              "both"      αν ρωτά και τα δύο
              null        αν is_relational == false
- source_law: {{"kind": "Ν." | "Π.Δ." | "Π.Ν.Π." | "Υ.Α." | "Α.Π.", "number": int, "year": int}} ή null
- relations: λίστα από {{"τροποποιεί","καταργεί","αντικαθιστά","προστίθεται","αναφέρεται"}} ή null για όλες

Παραδείγματα:
- "Ποιους νόμους τροποποιεί ο Ν. 5167/2024;" →
  {{"is_relational": true, "direction": "outgoing",
    "source_law": {{"kind":"Ν.","number":5167,"year":2024}},
    "relations": ["τροποποιεί"]}}
- "Ποιοι νόμοι κατήργησαν τον ν. 4001/2011;" →
  {{"is_relational": true, "direction": "incoming",
    "source_law": {{"kind":"Ν.","number":4001,"year":2011}},
    "relations": ["καταργεί"]}}
- "Πες μου για τη φορολογία ακινήτων το 2024" →
  {{"is_relational": false, "direction": null, "source_law": null, "relations": null}}
"""


_VALID_DIRECTIONS = {"outgoing", "incoming", "both"}


class RelationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("RelationAgent")
        self.llm = ChatOpenAI(
            model=settings.citation_llm_model,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}},
            api_key=settings.openai_api_key,
        )

    def _extract_intent(self, query: str) -> dict[str, Any]:
        try:
            resp = self.llm.invoke(
                [
                    SystemMessage(content=_INTENT_SYSTEM),
                    HumanMessage(content=_INTENT_USER_TEMPLATE.format(query=query)),
                ]
            )
            raw = resp.content or "{}"
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as e:
            _console.print(
                f"[yellow]RelationAgent intent extraction failed:[/yellow] {e}"
            )
            return {}

    def _resolve_focal_law(self, intent: dict[str, Any]) -> str | None:
        sl = intent.get("source_law")
        if not isinstance(sl, dict):
            return None
        kind = canonicalize_kind(str(sl.get("kind") or ""))
        if kind is None or kind not in CANONICAL_KINDS:
            return None
        try:
            number = int(sl.get("number"))
            year = int(sl.get("year"))
        except (TypeError, ValueError):
            return None
        return canonical_key(kind, number, year)

    def _validate_relations(self, intent: dict[str, Any]) -> list[str] | None:
        raw = intent.get("relations")
        if raw is None:
            return None
        if not isinstance(raw, list):
            return None
        out = [r for r in raw if isinstance(r, str) and r in RELATIONS]
        return out or None

    def _query_graph(
        self,
        focal_key: str,
        direction: str,
        relations: list[str] | None,
        limit: int,
    ) -> list[tuple[dict, str]]:
        """Return list of (citation_row, direction_label) tuples."""
        min_conf = settings.citation_min_confidence
        out: list[tuple[dict, str]] = []
        if direction in ("outgoing", "both"):
            rows = citations_from_law(focal_key, relations, min_conf, limit)
            for r in rows:
                out.append((r, "outgoing"))
        if direction in ("incoming", "both"):
            rows = citations_to_law(focal_key, relations, min_conf, limit)
            for r in rows:
                out.append((r, "incoming"))
        return out

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("rewritten_query") or state.get("query") or ""
        if not query:
            return {"relation_results": []}

        intent = self._extract_intent(query)
        if not intent.get("is_relational"):
            return {"relation_results": []}

        focal_key = self._resolve_focal_law(intent)
        if focal_key is None:
            return {"relation_results": []}

        direction = intent.get("direction") or "outgoing"
        if direction not in _VALID_DIRECTIONS:
            direction = "outgoing"
        relations = self._validate_relations(intent)
        limit = state.get("top_k") or settings.relation_top_k

        edges = self._query_graph(focal_key, direction, relations, limit)
        if not edges:
            return {"relation_results": []}

        # Hydrate citation rows into Document objects via their source chunk.
        chunk_ids = list({int(e[0]["source_chunk_id"]) for e in edges})
        chunk_rows = fetch_chunks_by_ids(chunk_ids)
        chunks_by_id = {int(r["id"]): r for r in chunk_rows}

        docs: list[Document] = []
        for citation, dir_label in edges:
            cid = int(citation["source_chunk_id"])
            row = chunks_by_id.get(cid)
            if row is None:
                continue
            base = {"id": cid, "content": row.get("content"), "metadata": row.get("metadata") or {}}
            doc = dict_to_document(base)
            if dir_label == "outgoing":
                source_key = focal_key
                target_key = citation.get("target_key")
            else:
                source_key = citation.get("source_key")
                target_key = focal_key
            doc.metadata["_relation_match"] = {
                "direction": dir_label,
                "source_law": source_key,
                "target_law": target_key,
                "target_article": citation.get("target_article"),
                "relation": citation.get("relation"),
                "snippet": citation.get("snippet"),
                "confidence": citation.get("confidence"),
            }
            docs.append(doc)
        return {"relation_results": docs}


@lru_cache(maxsize=1)
def get_relation_agent() -> RelationAgent:
    return RelationAgent()
