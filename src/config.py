from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str
    openai_chat_model: str = "gpt-5.4"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536

    metadata_llm_model: str = "gpt-5.4-mini"
    enable_llm_metadata: bool = True

    supabase_url: str
    supabase_service_key: str
    supabase_anon_key: str | None = None
    supabase_table: str = "documents"

    downloads_dir: Path = Path("downloads")

    enable_ocr: bool = True
    # Primary OCR engine; the other is the automatic fallback when the primary
    # errors or returns no text. Default flow: Mistral (cloud) -> Tesseract (local).
    ocr_engine: str = "mistral"  # "mistral" (cloud API) | "tesseract" (local)
    ocr_languages: str = "ell+eng"  # tesseract only
    mistral_api_key: str | None = None
    mistral_ocr_model: str = "mistral-ocr-latest"
    # PDFs larger than this skip Mistral's inline upload (it rejects ~>50MB) and
    # go straight to the Tesseract fallback instead of failing slowly.
    mistral_max_pdf_mb: int = 50
    min_text_chars: int = 500
    min_chars_per_page: int = 200
    min_page_coverage: float = 0.5

    chunk_tokens: int = 512

    top_k: int = 10
    hybrid_full_text_weight: float = 1.0
    hybrid_semantic_weight: float = 1.0
    rrf_k: int = 50

    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    rewriter_llm_model: str = "gpt-5.4-mini"
    enable_query_rewriter: bool = True

    enable_citation_extraction: bool = True
    citation_llm_model: str = "gpt-5.4-mini"
    citation_min_confidence: float = 0.5
    relation_top_k: int = 20

    checkpointer_dsn: str | None = None


settings = Settings()
