"""Shared LLM judges for the §5 end-to-end answer evals.

Discipline (best practice for LLM-as-judge):
  - pinned judge model, temperature 0, JSON mode
  - judge prompts live here in-repo (versioned with the metric)
  - judges return structured verdicts; the test layer aggregates + gates

The judge model is gpt-5.4-mini in JSON mode — the same configuration the routing
intent agents use successfully. Faithfulness/abstention are near-binary judgments,
so the mini model is adequate and cheap.
"""

from __future__ import annotations

import json
from functools import lru_cache

from openai import OpenAI

from src.config import settings

JUDGE_MODEL = settings.citation_llm_model  # gpt-5.4-mini


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _judge(system: str, user: str) -> dict:
    try:
        resp = _client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)[:200]}


# --------------------------------------------------------------------------- #
# Faithfulness / groundedness
# --------------------------------------------------------------------------- #

_FAITHFULNESS_SYSTEM = (
    "Είσαι αυστηρός αξιολογητής αξιοπιστίας απαντήσεων RAG για την ελληνική "
    "νομοθεσία. Σου δίνεται μια ΕΡΩΤΗΣΗ, μια ΑΠΑΝΤΗΣΗ και τα ΑΠΟΣΠΑΣΜΑΤΑ-ΠΗΓΕΣ "
    "που είχε διαθέσιμα ο βοηθός. Έργο σου: ανέλυσε την απάντηση σε επιμέρους "
    "ισχυρισμούς γεγονότων και έλεγξε ΑΝ ο καθένας υποστηρίζεται από τα "
    "αποσπάσματα. Μια άρνηση/δήλωση έλλειψης πληροφορίας ΔΕΝ είναι μη-υποστηριζόμενος "
    "ισχυρισμός (θεωρείται πλήρως αξιόπιστη). Επιστρέφεις ΜΟΝΟ JSON."
)

_FAITHFULNESS_USER = """ΕΡΩΤΗΣΗ:
{question}

ΑΠΑΝΤΗΣΗ:
{answer}

ΑΠΟΣΠΑΣΜΑΤΑ-ΠΗΓΕΣ:
{context}

Επέστρεψε JSON:
{{
  "total_claims": <ακέραιος>,
  "supported_claims": <ακέραιος>,
  "unsupported": [<σύντομη περιγραφή κάθε μη-υποστηριζόμενου ισχυρισμού>],
  "faithfulness": <0.0-1.0, = supported_claims/total_claims, ή 1.0 αν δεν υπάρχουν ισχυρισμοί>,
  "is_abstention": <true αν η απάντηση δηλώνει ρητά έλλειψη πληροφορίας>
}}"""


def judge_faithfulness(question: str, answer: str, context: str) -> dict:
    return _judge(
        _FAITHFULNESS_SYSTEM,
        _FAITHFULNESS_USER.format(question=question, answer=answer, context=context[:12000]),
    )


# --------------------------------------------------------------------------- #
# Abstention / refusal calibration
# --------------------------------------------------------------------------- #

_ABSTENTION_SYSTEM = (
    "Είσαι αξιολογητής. Σου δίνεται μια ΕΡΩΤΗΣΗ και η ΑΠΑΝΤΗΣΗ ενός βοηθού "
    "ελληνικής νομοθεσίας. Κρίνε ΜΟΝΟ ένα πράγμα: απάντησε ΟΥΣΙΑΣΤΙΚΑ στο "
    "συγκεκριμένο ερώτημα του χρήστη (έδωσε τη ζητούμενη πληροφορία), Ή αρνήθηκε/"
    "δήλωσε ότι η ζητούμενη πληροφορία δεν υπάρχει/δεν προκύπτει από τα διαθέσιμα "
    "αποσπάσματα;\n"
    "ΣΗΜΑΝΤΙΚΟ: Αν ο βοηθός λέει ότι το ζητούμενο (π.χ. ένας νόμος, μια ρύθμιση) "
    "ΔΕΝ προκύπτει / δεν αναφέρεται / δεν βρέθηκε στα αποσπάσματα, αυτό είναι "
    "ΑΡΝΗΣΗ (abstention) ΑΚΟΜΑ κι αν προσθέτει σχετικό πλαίσιο ή τι ΑΛΛΟ περιέχουν "
    "τα αποσπάσματα. Μετράει ως ουσιαστική απάντηση ΜΟΝΟ αν δίνει τη ζητούμενη "
    "συγκεκριμένη πληροφορία. Επιστρέφεις ΜΟΝΟ JSON."
)

_ABSTENTION_USER = """ΕΡΩΤΗΣΗ:
{query}

ΑΠΑΝΤΗΣΗ:
{answer}

Επέστρεψε JSON:
{{
  "answered_question": <true ΜΟΝΟ αν δόθηκε η ζητούμενη συγκεκριμένη πληροφορία·
                        false αν αρνήθηκε ή δήλωσε ότι δεν προκύπτει/δεν βρέθηκε>
}}"""


def judge_abstention(query: str, answer: str) -> dict:
    """Returns {'abstained': bool}. Abstained == did NOT answer the question."""
    v = _judge(_ABSTENTION_SYSTEM, _ABSTENTION_USER.format(query=query, answer=answer))
    if "_error" in v:
        return v
    return {"abstained": not bool(v.get("answered_question"))}
