from pathlib import Path

import ocrmypdf

from src.config import settings


def _cache_root() -> Path:
    return settings.downloads_dir / ".ocr"


def _cache_path(pdf: Path) -> Path:
    try:
        rel = pdf.resolve().relative_to(settings.downloads_dir.resolve())
    except ValueError:
        rel = Path(pdf.name)
    return _cache_root() / rel


def ensure_ocr(pdf: Path) -> Path:
    out = _cache_path(pdf)
    if out.exists() and out.stat().st_mtime >= pdf.stat().st_mtime:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    languages = [
        code.strip() for code in settings.ocr_languages.split("+") if code.strip()
    ]
    ocrmypdf.ocr(
        str(pdf),
        str(out),
        language=languages,
        force_ocr=True,
        deskew=True,
        rotate_pages=True,
        optimize=1,
        invalidate_digital_signatures=True,
        progress_bar=False,
    )
    return out
