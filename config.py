"""PulseDigest AI — typed application settings (Pydantic v2 / pydantic-settings).

Every secret is normalized on load: surrounding whitespace/quotes are stripped
and documented placeholder strings (`your_...`, `changeme`, `sk-xxx`, ...) are
collapsed to `None`. This prevents placeholder `.env` entries from being sent
upstream as bearer tokens, which providers answer with `401 User not found`.

The `.env` file is also treated as live configuration: `get_settings()` checks
its mtime/size on every call and rebuilds the cached `Settings` when the file
changed, so editing keys while uvicorn is running takes effect on the next
request without a restart.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root of this project (two levels up from src/ if packaged, else cwd)
PROJECT_ROOT = Path(__file__).resolve().parent

# `.env` is re-read from this path (absolute, so cwd never matters).
ENV_FILE: Path = PROJECT_ROOT / ".env"

# Values people paste into `.env.example` files. A secret containing any of
# these tokens is treated as "not configured" instead of being sent upstream.
PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "your_",
    "your-",
    "changeme",
    "change_me",
    "change-me",
    "replace_me",
    "replace-me",
    "insert_your",
    "placeholder",
    "sk-xxx",
    "tvly-xxx",
    "re_xxx",
    "example",
    "<",
    "xxx",
)

# Settings attributes holding provider API keys (single source of truth).
PROVIDER_API_KEY_FIELDS: tuple[str, ...] = (
    "openrouter_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "nvidia_nim_api_key",
    "deepseek_api_key",
    "groq_api_key",
    "custom_llm_api_key",
)

INTEGRATION_KEY_FIELDS: tuple[str, ...] = (
    "tavily_api_key",
    "resend_api_key",
    "discord_webhook_url",
)

ALL_SECRET_FIELDS: tuple[str, ...] = (*PROVIDER_API_KEY_FIELDS, *INTEGRATION_KEY_FIELDS)


# Public provider id → Settings attribute holding its API key
PROVIDER_KEY_FIELD_BY_NAME: dict[str, str] = {
    "openrouter": "openrouter_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "google": "google_api_key",
    "nvidia_nim": "nvidia_nim_api_key",
    "deepseek": "deepseek_api_key",
    "groq": "groq_api_key",
    "custom": "custom_llm_api_key",
}


def is_placeholder(value: str | None) -> bool:
    """True when `value` looks like a documentation placeholder, not a secret."""
    if not value:
        return False
    lowered = value.lower()
    return any(token in lowered for token in PLACEHOLDER_TOKENS)


def sanitize_secret(value: object | None) -> str | None:
    """Strip whitespace/quotes from a secret and reject placeholders.

    Returns `None` for empty strings, placeholder strings and non-strings that
    stringify to something empty. Real credentials are returned verbatim
    (quotes and surrounding whitespace removed).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip().strip("'\"").strip()
    if not cleaned:
        return None
    if is_placeholder(cleaned):
        return None
    return cleaned


def sanitize_optional_text(value: object | None) -> str | None:
    """Strip whitespace/quotes from an optional non-secret string (e.g. URL)."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip().strip("'\"").strip()
    return cleaned or None


def mask_secret(value: str | None, head: int = 7, tail: int = 4) -> str:
    """Return a safe fingerprint such as `sk-or-v…4a2b` (never the full key)."""
    if not value:
        return ""
    if len(value) <= head + tail:
        return "•" * max(4, min(len(value), 8))
    return f"{value[:head]}…{value[-tail:]}"


class Settings(BaseSettings):
    """Central runtime settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Global LLM defaults ---
    default_llm_provider: str = "openrouter"
    default_llm_model: str = Field(
        default="openrouter/auto",
        description="Fallback model; never hardcode deprecated identifiers.",
    )

    # --- Cloud LLM provider API keys ---
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    nvidia_nim_api_key: str | None = None
    deepseek_api_key: str | None = None
    groq_api_key: str | None = None
    custom_llm_api_key: str | None = None
    custom_llm_base_url: str | None = None

    # --- External integrations (graceful degradation when empty) ---
    tavily_api_key: str | None = None
    resend_api_key: str | None = None
    resend_from_email: str | None = None
    discord_webhook_url: str | None = None

    # --- App settings ---
    db_path: str = "pulsedigest.db"
    http_timeout_seconds: float = 30.0

    @field_validator(*ALL_SECRET_FIELDS, mode="before")
    @classmethod
    def _clean_secret(cls, value: object | None) -> str | None:
        """Strip quotes/whitespace and drop placeholder values."""
        return sanitize_secret(value)

    @field_validator("resend_from_email", "custom_llm_base_url", mode="before")
    @classmethod
    def _clean_text(cls, value: object | None) -> str | None:
        """Normalize non-secret optional strings."""
        return sanitize_optional_text(value)

    @field_validator("default_llm_model", mode="before")
    @classmethod
    def _clean_model(cls, value: object | None) -> str:
        """Never allow a blank default model name."""
        return sanitize_optional_text(value) or "openrouter/auto"

    def configured_providers(self) -> dict[str, bool]:
        """Provider → bool map without exposing any secret value."""
        return {
            provider: bool(getattr(self, field_name, None))
            for provider, field_name in PROVIDER_KEY_FIELD_BY_NAME.items()
        }

    def env_status(self) -> dict[str, object]:
        """Public configuration snapshot for the dashboard badges (no secrets)."""
        return {
            "tavily_configured": bool(self.tavily_api_key),
            "resend_configured": bool(self.resend_api_key and self.resend_from_email),
            "discord_configured": bool(self.discord_webhook_url),
            "providers_configured": self.configured_providers(),
        }

    def provider_key_fingerprints(self) -> dict[str, str]:
        """Provider → masked key fingerprint for configured `.env` keys only.

        A fingerprint (`sk-or-v…4a2b`) lets the dashboard show which key is in
        use without ever exposing the secret itself.
        """
        fingerprints: dict[str, str] = {}
        for provider, field_name in PROVIDER_KEY_FIELD_BY_NAME.items():
            fingerprint = mask_secret(getattr(self, field_name, None))
            if fingerprint:
                fingerprints[provider] = fingerprint
        return fingerprints


_settings: Settings | None = None
_settings_signature: tuple[str, int | None, int | None] | None = None


def _env_signature(env_file: Path | str | None = None) -> tuple[str, int | None, int | None]:
    """Cheap change-detection token for the `.env` file (mtime + size)."""
    path = Path(env_file) if env_file is not None else ENV_FILE
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), None, None)


def get_settings() -> Settings:
    """Return `Settings`, re-reading `.env` whenever the file changed on disk.

    The check is a single `stat()` per call, so a long-running server picks up
    newly pasted keys immediately without a restart (and without clobbering
    real process environment variables, which still take precedence).
    """
    global _settings, _settings_signature
    signature = _env_signature()
    if _settings is None or signature != _settings_signature:
        _settings = Settings(_env_file=str(ENV_FILE))
        _settings_signature = signature
    return _settings


def reload_settings() -> Settings:
    """Force a fresh `.env` parse (used by `/api/v1/providers` and tests)."""
    global _settings, _settings_signature
    _settings = None
    _settings_signature = None
    return get_settings()


def reset_settings() -> None:
    """Drop the cached settings (used by tests to reload env vars)."""
    global _settings, _settings_signature
    _settings = None
    _settings_signature = None


settings = get_settings()
