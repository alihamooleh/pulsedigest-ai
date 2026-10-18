"""PulseDigest AI — Pydantic v2 request/response/data schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from config import sanitize_optional_text, sanitize_secret


class LLMProviderEnum(str, Enum):
    """Supported cloud LLM providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    NVIDIA_NIM = "nvidia_nim"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    CUSTOM = "custom"


class LLMConfigOverride(BaseModel):
    """Per-request LLM provider override."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProviderEnum
    model_name: str = Field(..., min_length=1)
    custom_base_url: str | None = None
    api_key_override: str | None = None

    @field_validator("model_name")
    @classmethod
    def model_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be blank")
        return v.strip()

    @field_validator("custom_base_url", mode="before")
    @classmethod
    def base_url_https(cls, v: object | None) -> str | None:
        cleaned = sanitize_optional_text(v)
        if cleaned is not None:
            cleaned = cleaned.rstrip("/")
            if cleaned and not cleaned.startswith(("http://", "https://")):
                raise ValueError("custom_base_url must start with http:// or https://")
        return cleaned or None


class IntegrationsConfigOverride(BaseModel):
    """Per-request overrides for Tavily / Resend / Discord credentials.

    Lets the browser vault supply keys without redeploying the server. Every
    value is cleaned like a `.env` secret: quotes/whitespace stripped and
    placeholder strings (`your_...`, `tvly-xxx`, `example`) collapsed to `None`.
    """

    model_config = ConfigDict(extra="forbid")

    tavily_api_key: str | None = None
    resend_api_key: str | None = None
    resend_from_email: str | None = None
    discord_webhook_url: str | None = None

    @field_validator("tavily_api_key", "resend_api_key", mode="before")
    @classmethod
    def clean_secret(cls, v: object | None) -> str | None:
        return sanitize_secret(v)

    @field_validator("resend_from_email", "discord_webhook_url", mode="before")
    @classmethod
    def clean_text(cls, v: object | None) -> str | None:
        return sanitize_optional_text(v)

    def empty(self) -> bool:
        """True when no integration override was supplied."""
        return not any(self.model_dump().values())


class DigestRequest(BaseModel):
    """Inbound digest research request."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=3)
    # "auto" mirrors the topic language; any explicit name/code (Persian, fa,
    # English, ...) forces that target language for the whole report.
    output_language: str = Field(default="auto", max_length=40)
    recipient_email: EmailStr | None = None
    notify_discord: bool = False
    notify_email: bool = False
    llm_config: LLMConfigOverride | None = None

    # Integration credentials: nested object and/or flat convenience fields.
    integrations_config: IntegrationsConfigOverride | None = None
    tavily_api_key: str | None = None
    resend_api_key: str | None = None
    resend_from_email: str | None = None
    discord_webhook_url: str | None = None

    @field_validator("topic")
    @classmethod
    def topic_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be blank")
        return v.strip()

    @field_validator("output_language", mode="before")
    @classmethod
    def clean_output_language(cls, v: object | None) -> str:
        """Blank/whitespace/quoted values fall back to language mirroring."""
        return sanitize_optional_text(v) or "auto"

    @field_validator("tavily_api_key", "resend_api_key", mode="before")
    @classmethod
    def clean_secret(cls, v: object | None) -> str | None:
        return sanitize_secret(v)

    @field_validator("resend_from_email", "discord_webhook_url", mode="before")
    @classmethod
    def clean_text(cls, v: object | None) -> str | None:
        return sanitize_optional_text(v)

    def integration_overrides(self) -> dict[str, str | None]:
        """Merged UI overrides: nested config first, flat fields win."""
        nested = (
            self.integrations_config.model_dump()
            if self.integrations_config is not None
            else {}
        )
        resolved: dict[str, str | None] = {
            "tavily_api_key": nested.get("tavily_api_key"),
            "resend_api_key": nested.get("resend_api_key"),
            "resend_from_email": nested.get("resend_from_email"),
            "discord_webhook_url": nested.get("discord_webhook_url"),
        }
        for key, value in self.model_dump(include=set(resolved)).items():
            if value:
                resolved[key] = value
        return resolved


class NotificationStatus(BaseModel):
    """Per-channel dispatch outcome (graceful degradation friendly)."""

    sent: bool = False
    detail: str | None = None


class DigestResponse(BaseModel):
    """Outbound digest result."""

    id: int
    topic: str
    report_markdown: str
    provider: str
    model_name: str
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int | None = None
    notifications: dict[str, NotificationStatus] = Field(default_factory=dict)
    created_at: datetime


class DependencyUpdateStatus(BaseModel):
    """Self-update watchdog report for a single package."""

    package_name: str
    installed_version: str
    latest_stable_version: str
    update_available: bool
    release_url: str


# --- Live Model Discovery (core feature) ---


class ModelDiscoveryRequest(BaseModel):
    """Request to discover the live, non-deprecated model list of a provider."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProviderEnum = LLMProviderEnum.OPENROUTER
    api_key_override: str | None = None
    custom_base_url: str | None = None

    @field_validator("custom_base_url", mode="before")
    @classmethod
    def base_url_https(cls, v: object | None) -> str | None:
        cleaned = sanitize_optional_text(v)
        if cleaned is not None:
            cleaned = cleaned.rstrip("/")
            if cleaned and not cleaned.startswith(("http://", "https://")):
                raise ValueError("custom_base_url must start with http:// or https://")
        return cleaned or None


class ModelInfo(BaseModel):
    """A single model returned by a provider's live endpoint."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    provider: LLMProviderEnum
    context_length: int | None = None


class ModelDiscoveryResponse(BaseModel):
    """Provider plus its live model list."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProviderEnum
    models: list[ModelInfo] = Field(default_factory=list)


def schema_to_dict(model: BaseModel) -> dict[str, Any]:
    """Pydantic v2 serialization helper (never use deprecated `.dict()`)."""
    return model.model_dump()
