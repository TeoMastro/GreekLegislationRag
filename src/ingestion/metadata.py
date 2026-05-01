import json
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from rich.console import Console

from src.config import settings


_console = Console()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


_METADATA_SYSTEM = (
    "Είσαι ειδικός στην ελληνική νομοθεσία. Εξάγεις μεταδεδομένα από κείμενο ΦΕΚ "
    "και επιστρέφεις ΜΟΝΟ έγκυρο JSON. Αν κάποιο πεδίο δεν είναι σαφές, χρησιμοποίησε null."
)

_METADATA_USER_TEMPLATE = (
    "Από το παρακάτω απόσπασμα ΦΕΚ, εξάγαγε μεταδεδομένα ως JSON με τα κλειδιά:\n"
    "title (σύντομος τίτλος), fek_number (π.χ. '123/Α/2024'), publication_date (YYYY-MM-DD), "
    "doc_type (π.χ. 'Νόμος', 'Προεδρικό Διάταγμα', 'Υπουργική Απόφαση'), "
    "authority (η εκδούσα αρχή), subject (1-2 πρόταση περίληψη του θέματος).\n\n"
    "Κείμενο:\n{excerpt}"
)


def metadata_from_path(pdf: Path) -> dict:
    name = pdf.stem
    base = {"source": pdf.name, "source_path": str(pdf)}

    year: int | None = None
    try:
        year = int(pdf.parent.name)
    except ValueError:
        if len(name) >= 4 and name[:4].isdigit():
            year = int(name[:4])
    if year is not None:
        base["year"] = year

    if len(name) >= 11 and name.isdigit():
        base["fek_series_code"] = name[4:7]
        base["sequence"] = name[7:11]
    return base


def enrich_with_llm(text: str) -> dict:
    if not settings.enable_llm_metadata:
        return {}
    excerpt = text[:4000]
    if not excerpt.strip():
        return {}
    try:
        resp = _client().chat.completions.create(
            model=settings.metadata_llm_model,
            messages=[
                {"role": "system", "content": _METADATA_SYSTEM},
                {"role": "user", "content": _METADATA_USER_TEMPLATE.format(excerpt=excerpt)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as e:
        _console.print(f"[yellow]LLM metadata enrichment failed:[/yellow] {e}")
        return {}
    return {k: v for k, v in data.items() if v not in (None, "", [])}
