"""PulseDigest AI — universal cloud LLM factory + Live Model Discovery.

Supports 8 providers through OpenAI-compatible `base_url` routing
(Anthropic via its OpenAI compatibility layer) and queries each provider's
live `/models` endpoint to return currently-available, non-deprecated models.
The default fallback model is `openrouter/auto` — never hardcoded, deprecated ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import (
    PROVIDER_KEY_FIELD_BY_NAME,
    Settings,
    get_settings,
    mask_secret,
    sanitize_optional_text,
    sanitize_secret,
)
from schemas import LLMConfigOverride, LLMProviderEnum, ModelInfo

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"

# OpenRouter attribution headers (required for app rankings, useful in logs).
OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/pulsedigest-ai",
    "X-Title": "PulseDigest AI",
}

# OpenRouter's public `/models` list answers 200 even for invalid tokens, so
# discovery additionally verifies the key against this authenticated endpoint.
OPENROUTER_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"

# OpenAI-compatible base URLs per provider (TRD section 2)
OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    LLMProviderEnum.OPENROUTER.value: "https://openrouter.ai/api/v1",
    LLMProviderEnum.OPENAI.value: "https://api.openai.com/v1",
    LLMProviderEnum.GOOGLE.value: "https://generativelanguage.googleapis.com/v1beta/openai",
    LLMProviderEnum.NVIDIA_NIM.value: "https://integrate.api.nvidia.com/v1",
    LLMProviderEnum.DEEPSEEK.value: "https://api.deepseek.com/v1",
    LLMProviderEnum.GROQ.value: "https://api.groq.com/openai/v1",
    # CUSTOM is resolved dynamically from custom_base_url
}

# Which settings attribute holds each provider's API key
PROVIDER_KEY_ATTRIBUTES: dict[str, str] = dict(PROVIDER_KEY_FIELD_BY_NAME)

# Keys each provider returns for context window size
_CONTEXT_LENGTH_KEYS = ("context_length", "context_window", "context_size")


def is_openai_compatible(provider: str | LLMProviderEnum) -> bool:
    """True unless the provider needs its native protocol (Anthropic)."""
    return str(getattr(provider, "value", provider)) != LLMProviderEnum.ANTHROPIC.value


class ProviderAuthError(ValueError):
    """Raised when a provider API key is missing or invalid."""


@dataclass(frozen=True)
class KeyResolution:
    """Effective API key plus where it came from (for transparent UI badges)."""

    api_key: str | None
    source: str  # "override" | "env" | "missing"

    @property
    def source_label(self) -> str:
        return {"override": "UI Override", "env": "Server .env"}.get(
            self.source, "Missing key"
        )

    @property
    def fingerprint(self) -> str:
        """Masked key fingerprint, e.g. `sk-or-v…4a2b`."""
        return mask_secret(self.api_key)

    def describe(self) -> str:
        if not self.api_key:
            return "missing key (neither UI Override nor Server .env)"
        return f"{self.source_label} ({self.fingerprint})"


def describe_api_key(
    provider: str | LLMProviderEnum,
    settings: Settings,
    api_key_override: str | None = None,
) -> KeyResolution:
    """Resolve the effective key and its source (UI override → `.env`).

    A blank/placeholder override (e.g. `your_openai_key`) never masks a real
    `.env` key; it falls through, which mirrors `resolve_api_key` exactly.
    """
    provider_str = str(getattr(provider, "value", provider))
    override = sanitize_secret(api_key_override)
    if override:
        return KeyResolution(override, "override")
    field_name = PROVIDER_KEY_ATTRIBUTES.get(provider_str)
    env_value = (
        sanitize_secret(getattr(settings, field_name, None)) if field_name else None
    )
    return KeyResolution(env_value, "env" if env_value else "missing")


def resolve_api_key(
    provider: str | LLMProviderEnum,
    settings: Settings,
    api_key_override: str | None = None,
) -> str | None:
    """Return the effective API key for `provider`.

    Precedence: sanitized `api_key_override` → sanitized `.env` value. A blank
    or placeholder override (e.g. `your_openai_key`) falls back to `.env`
    instead of being sent upstream and triggering `401 User not found`.
    """
    return describe_api_key(provider, settings, api_key_override).api_key


def _resolve_provider_parts(
    provider: str | LLMProviderEnum,
    settings: Settings,
    api_key_override: str | None = None,
    custom_base_url: str | None = None,
) -> tuple[str, str, str | None]:
    """Return `(provider_str, api_key, base_url_override)` normalized."""
    provider_str = str(getattr(provider, "value", provider))
    if provider_str not in PROVIDER_KEY_ATTRIBUTES:
        raise ValueError(f"Unsupported provider: {provider_str!r}")

    # UI/per-request key strictly wins; sanitized so "sk-..." stays clean and
    # placeholders (`your_...`) never mask a real `.env` key.
    key_info = describe_api_key(provider_str, settings, api_key_override)
    api_key = key_info.api_key or ""
    if not api_key:
        raise ProviderAuthError(
            f"No API key available for provider {provider_str!r} — paste one in the "
            "vault, add it to .env, or pass api_key_override."
        )

    base_url_override = sanitize_optional_text(custom_base_url)
    if provider_str == LLMProviderEnum.CUSTOM.value:
        base_url_override = base_url_override or settings.custom_llm_base_url
        if not base_url_override:
            raise ValueError(
                "Provider 'custom' requires a base URL (custom_base_url or CUSTOM_LLM_BASE_URL env var)."
            )
    return provider_str, api_key, base_url_override


def _resolve_model_name(
    config_override: LLMConfigOverride | None, settings: Settings
) -> str:
    """Resolve the model name from override or the `openrouter/auto` default."""
    if config_override is not None and config_override.model_name:
        return config_override.model_name
    return settings.default_llm_model


def create_chat_model(
    config_override: LLMConfigOverride | None = None,
    settings: Settings | None = None,
    http_client: httpx.Client | None = None,
) -> BaseChatModel:
    """Build a LangChain chat model for any of the 8 supported providers.

    Configuration precedence: `config_override` → provider defaults from
    `settings`. Raises `ProviderAuthError`/`ValueError` on missing config.
    """
    settings = settings or get_settings()
    if config_override is None:
        provider_str = settings.default_llm_provider
        if provider_str not in PROVIDER_KEY_ATTRIBUTES:
            raise ValueError(f"Unsupported provider: {provider_str!r}")
        api_key = resolve_api_key(provider_str, settings) or ""
        base_url_override: str | None = (
            settings.custom_llm_base_url
            if provider_str == LLMProviderEnum.CUSTOM.value
            else None
        )
        if not api_key:
            raise ProviderAuthError(
                f"No API key available for provider {provider_str!r} in settings."
            )
        model_name = _resolve_model_name(None, settings)
    else:
        provider_str, api_key, base_url_override = _resolve_provider_parts(
            config_override.provider, settings,
            api_key_override=config_override.api_key_override,
            custom_base_url=config_override.custom_base_url,
        )
        model_name = _resolve_model_name(config_override, settings)

    if is_openai_compatible(provider_str):
        base_url = base_url_override or OPENAI_COMPATIBLE_BASE_URLS.get(provider_str)
        if not base_url:
            raise ValueError(f"No base URL resolvable for provider {provider_str!r}")
        kwargs: dict = {"model": model_name, "api_key": api_key, "base_url": base_url}
        if provider_str == LLMProviderEnum.OPENROUTER.value:
            kwargs["default_headers"] = dict(OPENROUTER_HEADERS)
        if http_client is not None:  # pragma: no cover - injected in tests
            kwargs["http_client"] = http_client
        return ChatOpenAI(**kwargs)

    # Anthropic: use its OpenAI compatibility layer so all providers share one client type.
    # Native auth (x-api-key) is only needed for /models discovery below.
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=f"{ANTHROPIC_BASE_URL}/",
    )


async def fetch_available_models(
    provider: str | LLMProviderEnum,
    api_key_override: str | None = None,
    custom_base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> list[ModelInfo]:
    """Query a provider's live `/models` endpoint and return non-deprecated models.

    - OpenAI-compatible providers: `GET {base_url}/models` with
      `Authorization: Bearer <key>` via `httpx.AsyncClient`.
    - Anthropic: `GET https://api.anthropic.com/v1/models` with
      `x-api-key` + `anthropic-version` headers.

    Raises `ProviderAuthError` on missing keys and `ValueError` on
    unreachable endpoints / invalid API keys.
    """
    settings = settings or get_settings()
    provider_str, api_key, base_url_override = _resolve_provider_parts(
        provider, settings,
        api_key_override=api_key_override,
        custom_base_url=custom_base_url,
    )

    key_info = describe_api_key(provider_str, settings, api_key_override)

    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=settings.http_timeout_seconds)
    try:
        if provider_str == LLMProviderEnum.OPENROUTER.value:
            # `/models` is public and ignores bad tokens, so verify the key
            # first — otherwise an invalid key only fails later in the digest.
            await _verify_openrouter_key(client, key_info)
        return await _query_models_list(
            client, provider_str, api_key, base_url_override, key_info
        )
    finally:
        if owned_client:
            await client.aclose()


async def _verify_openrouter_key(
    client: httpx.AsyncClient, key_info: KeyResolution
) -> None:
    """Validate an OpenRouter key via `GET /api/v1/auth/key` (masked errors)."""
    try:
        response = await client.get(
            OPENROUTER_AUTH_KEY_URL,
            headers={"Authorization": f"Bearer {key_info.api_key}"},
        )
    except httpx.HTTPError as exc:
        raise ValueError(
            f"Could not reach the OpenRouter auth endpoint ({OPENROUTER_AUTH_KEY_URL}): {exc}"
        ) from exc

    if response.status_code in (401, 403):
        if key_info.source == "override":
            remedy = (
                "Paste the replacement key in the key box (it replaces the UI "
                "override), or click '× Use server .env instead' if the .env key is "
                "the valid one."
            )
        else:
            remedy = (
                "Update OPENROUTER_API_KEY in .env — it is re-read live, no uvicorn "
                "restart needed — or paste the new key in the key box."
            )
        raise ProviderAuthError(
            f"OpenRouter rejected the API key from {key_info.describe()} "
            f"(HTTP {response.status_code}). Create a fresh key at "
            f"https://openrouter.ai/keys. {remedy}"
        )
    if response.status_code >= 400:
        raise ValueError(
            "OpenRouter auth check returned HTTP "
            f"{response.status_code}: {response.text[:200]}"
        )


def _is_deprecated(model_id: str, raw: dict) -> bool:
    """Heuristic filter for deprecated / retired models."""
    lowered = model_id.lower()
    if "deprecated" in lowered or ":deactivated" in lowered:
        return True
    return str(raw.get("status", "")).lower() in {"deprecated", "retired"}


async def _query_models_list(
    client: httpx.AsyncClient,
    provider_str: str,
    api_key: str,
    base_url_override: str | None,
    key_info: KeyResolution | None = None,
) -> list[ModelInfo]:
    """Fetch and parse the live model list for one provider."""
    provider_enum = LLMProviderEnum(provider_str)

    if provider_enum == LLMProviderEnum.ANTHROPIC:
        request_url = ANTHROPIC_MODELS_URL
        headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    else:
        base = base_url_override or OPENAI_COMPATIBLE_BASE_URLS.get(provider_str)
        if not base:
            raise ValueError(f"No base URL resolvable for provider {provider_str!r}")
        request_url = f"{base.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}

    logger.info("Discovering live models: %s", request_url)
    try:
        response = await client.get(request_url, headers=headers)
    except httpx.HTTPError as exc:
        raise ValueError(
            f"Failed to reach {provider_str} models endpoint ({request_url}): {exc}"
        ) from exc

    if response.status_code == 401:
        key_detail = key_info.describe() if key_info else "unknown key"
        raise ProviderAuthError(
            f"Invalid API key for provider {provider_str!r} (401). "
            f"Key in use: {key_detail}."
        )
    if response.status_code >= 400:
        raise ValueError(
            f"{provider_str} models endpoint returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"{provider_str} models endpoint returned non-JSON payload."
        ) from exc

    items: list[dict] = (
        payload.get("data") if isinstance(payload.get("data"), list) else payload.get("models")
        or []
    )

    models: list[ModelInfo] = []
    for raw in items:
        model_id = str(raw.get("id") or raw.get("name") or "")
        if not model_id or _is_deprecated(model_id, raw):
            continue
        context_length = next(
            (
                int(raw[key])
                for key in _CONTEXT_LENGTH_KEYS
                if isinstance(raw.get(key), int)
            ),
            None,
        )
        if provider_enum == LLMProviderEnum.OPENROUTER and context_length is None:
            top_provider = raw.get("top_provider") or {}
            if isinstance(top_provider.get("context_length"), int):
                context_length = int(top_provider["context_length"])
        display_name = (
            raw.get("display_name")
            or raw.get("name")
            or raw.get("displayName")
            or model_id
        )
        models.append(
            ModelInfo(
                id=model_id,
                name=str(display_name),
                provider=provider_enum,
                context_length=context_length,
            )
        )
    return models
