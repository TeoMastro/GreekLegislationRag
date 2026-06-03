"""Root conftest.

Two jobs:
  1. Make the repo root importable so `import src...` resolves when pytest
     collects tests under evals/.
  2. Let the deterministic component evals construct `src.config.settings`
     WITHOUT real credentials (e.g. in CI with no .env), by injecting placeholder
     secrets — but ONLY when no real config is available.

Critical subtlety: pydantic-settings ranks real environment variables ABOVE
.env-file values. So we must NOT blindly seed os.environ with placeholders — on a
dev machine the real keys live in .env (not os.environ), and a placeholder in
os.environ would override them (401s on every online eval). We therefore probe
whether settings can be built from real sources first, and only fall back to
placeholders when that fails.
"""

import os

_PLACEHOLDERS = {
    "OPENAI_API_KEY": "sk-test-placeholder",
    "SUPABASE_URL": "https://placeholder.supabase.co",
    "SUPABASE_SERVICE_KEY": "test-placeholder-service-key",
}


def _real_config_available() -> bool:
    """True if src.config.settings builds from real env + .env sources."""
    try:
        # Importing runs the module-level `settings = Settings()`, which reads
        # env vars and .env. Success => real config exists; failure (missing
        # required fields) => fall back to placeholders below. A failed import is
        # evicted from sys.modules, so the later real import retries cleanly.
        import src.config  # noqa: F401

        return True
    except Exception:
        return False


if not _real_config_available():
    for _k, _v in _PLACEHOLDERS.items():
        os.environ.setdefault(_k, _v)
