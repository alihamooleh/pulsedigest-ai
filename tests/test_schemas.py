"""Unit tests for Pydantic v2 schemas (`schemas.py`)."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from schemas import (
    DependencyUpdateStatus,
    DigestRequest,
    DigestResponse,
    IntegrationsConfigOverride,
    LLMConfigOverride,
    LLMProviderEnum,
    ModelDiscoveryRequest,
    ModelDiscoveryResponse,
    ModelInfo,
    NotificationStatus,
    schema_to_dict,
)


class TestDigestRequest:
    def test_valid_request(self) -> None:
        req = DigestRequest(topic="AI in healthcare", notify_discord=True)
        assert req.topic == "AI in healthcare"
        assert req.recipient_email is None
        assert req.notify_email is False

    def test_valid_with_email(self) -> None:
        req = DigestRequest(
            topic="quantum computing",
            recipient_email="user@example.com",
            notify_email=True,
            llm_config=LLMConfigOverride(
                provider=LLMProviderEnum.OPENAI, model_name="gpt-5.6"
            ),
        )
        assert req.recipient_email == "user@example.com"
        assert req.llm_config is not None
        assert req.llm_config.provider == LLMProviderEnum.OPENAI

    def test_topic_too_short(self) -> None:
        with pytest.raises(ValidationError):
            DigestRequest(topic="ab")

    def test_topic_blank_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DigestRequest(topic="   ")

    def test_invalid_email_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DigestRequest(topic="valid topic", recipient_email="not-an-email")

    @pytest.mark.parametrize(
        "provider",
        [
            "openrouter",
            "openai",
            "anthropic",
            "google",
            "nvidia_nim",
            "deepseek",
            "groq",
            "custom",
        ],
    )
    def test_all_eight_providers_accepted(self, provider: str) -> None:
        req = DigestRequest(
            topic="valid topic",
            llm_config=LLMConfigOverride(provider=provider, model_name="m"),
        )
        assert req.llm_config is not None
        assert req.llm_config.provider.value == provider

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigOverride(provider="ollama", model_name="local-1")


class TestOutputLanguage:
    """Task 2: auto mirrors the topic; explicit values force a target language."""

    def test_defaults_to_auto(self) -> None:
        assert DigestRequest(topic="AI agents").output_language == "auto"

    @pytest.mark.parametrize("value", ["", "   ", "'auto'"])
    def test_blank_or_quoted_auto_normalizes(self, value: str) -> None:
        assert DigestRequest(topic="AI agents", output_language=value).output_language == "auto"

    @pytest.mark.parametrize("value", ["Persian", "English", "Arabic", "German", "French", "fa"])
    def test_explicit_language_is_preserved(self, value: str) -> None:
        request = DigestRequest(topic="AI agents", output_language=f"  {value}  ")
        assert request.output_language == value

    def test_overlong_language_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DigestRequest(topic="AI agents", output_language="x" * 41)


class TestLLMConfigOverride:
    def test_base_url_validation(self) -> None:
        ok = LLMConfigOverride(
            provider=LLMProviderEnum.CUSTOM,
            model_name="my-model",
            custom_base_url="https://api.example.com/v1/",
        )
        assert ok.custom_base_url == "https://api.example.com/v1"

        with pytest.raises(ValidationError):
            LLMConfigOverride(
                provider=LLMProviderEnum.CUSTOM,
                model_name="my-model",
                custom_base_url="ftp://bad.example.com",
            )

    def test_blank_model_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigOverride(provider=LLMProviderEnum.OPENAI, model_name="  ")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigOverride(provider=LLMProviderEnum.OPENAI, model_name="m", rogue=1)


class TestModelDiscoverySchemas:
    def test_discovery_request_defaults_and_digest_sealed(self) -> None:
        req = ModelDiscoveryRequest()
        assert req.provider == LLMProviderEnum.OPENROUTER
        assert req.api_key_override is None
        assert req.custom_base_url is None
        with pytest.raises(ValidationError):
            ModelDiscoveryRequest(provider=None)

    def test_discovery_request_custom_base_url_validated(self) -> None:
        req = ModelDiscoveryRequest(
            provider="custom", custom_base_url=" https://api.custom.dev/v1 "
        )
        assert req.custom_base_url == "https://api.custom.dev/v1"
        with pytest.raises(ValidationError):
            ModelDiscoveryRequest(provider="custom", custom_base_url="nope")

    def test_model_info(self) -> None:
        info = ModelInfo(
            id="openrouter/auto",
            name="Auto Router",
            provider="openrouter",
            context_length=200000,
        )
        assert info.provider == LLMProviderEnum.OPENROUTER
        assert info.context_length == 200000
        assert info.model_dump() == {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "provider": LLMProviderEnum.OPENROUTER,
            "context_length": 200000,
        }

    def test_discovery_response_holds_model_list(self) -> None:
        resp = ModelDiscoveryResponse(
            provider="openai",
            models=[
                ModelInfo(id="gpt-5.6", name="GPT 5.6", provider="openai"),
                ModelInfo(
                    id="gpt-6", name="GPT 6", provider="openai", context_length=1_000_000
                ),
            ],
        )
        assert len(resp.models) == 2
        dumped = schema_to_dict(resp)
        assert dumped["models"][1]["id"] == "gpt-6"
        assert dumped["models"][0]["provider"] == LLMProviderEnum.OPENAI.value


class TestOtherSchemas:
    def test_notification_status_defaults_to_unsent(self) -> None:
        assert NotificationStatus().sent is False

    def test_digest_response_round_trip(self) -> None:
        resp = DigestResponse(
            id=1,
            topic="edge ai",
            report_markdown="# Digest",
            provider="openrouter",
            model_name="openrouter/auto",
            tokens_used=123,
            estimated_cost_usd=0.004,
            notifications={"discord": NotificationStatus(sent=True)},
            created_at=datetime(2026, 10, 16, 12, 0, 0),
        )
        data = resp.model_dump()
        assert data["model_name"] == "openrouter/auto"
        assert data["notifications"]["discord"]["sent"] is True

    def test_dependency_update_status(self) -> None:
        status = DependencyUpdateStatus(
            package_name="fastapi",
            installed_version="0.143.2",
            latest_stable_version="0.144.0",
            update_available=True,
            release_url="https://pypi.org/project/fastapi/",
        )
        assert status.update_available is True

    def test_digest_response_accepts_latency(self) -> None:
        resp = DigestResponse(
            id=2,
            topic="latency",
            report_markdown="# x",
            provider="groq",
            model_name="llama-4",
            latency_ms=1234,
            created_at=datetime(2026, 10, 18, 9, 0, 0),
        )
        assert resp.latency_ms == 1234
        assert DigestResponse(
            id=3,
            topic="no latency",
            report_markdown="# y",
            provider="groq",
            model_name="llama-4",
            created_at=datetime(2026, 10, 18, 9, 0, 0),
        ).latency_ms is None


class TestIntegrationsConfigOverride:
    """Task 2: per-request Tavily / Resend / Discord credential overrides."""

    def test_strips_quotes_and_whitespace(self) -> None:
        cfg = IntegrationsConfigOverride(
            tavily_api_key='  "tvly-live-123"  ',
            resend_api_key="'re_abc'",
            resend_from_email="  digest@firm.io ",
            discord_webhook_url=' "https://discord.com/api/webhooks/1/x" ',
        )
        assert cfg.tavily_api_key == "tvly-live-123"
        assert cfg.resend_api_key == "re_abc"
        assert cfg.resend_from_email == "digest@firm.io"
        assert cfg.discord_webhook_url == "https://discord.com/api/webhooks/1/x"

    @pytest.mark.parametrize(
        "value",
        [
            "your_tavily_key",
            "your-resend-key",
            "changeme",
            "change_me",
            "sk-xxx",
            "tvly-xxx",
            "re_xxx",
            "key-example-here",
            "   ",
            "",
        ],
    )
    def test_placeholder_and_blank_keys_rejected(self, value: str) -> None:
        cfg = IntegrationsConfigOverride(
            tavily_api_key=value,
            resend_api_key=value,
            resend_from_email="  ",
            discord_webhook_url="  ",
        )
        assert cfg.tavily_api_key is None
        assert cfg.resend_api_key is None
        assert cfg.resend_from_email is None
        assert cfg.discord_webhook_url is None
        assert cfg.empty() is True

    def test_real_looking_keys_are_kept(self) -> None:
        cfg = IntegrationsConfigOverride(tavily_api_key="tvly-dev-abc123", resend_api_key="re_K9x2")
        assert cfg.tavily_api_key == "tvly-dev-abc123"
        assert cfg.resend_api_key == "re_K9x2"
        assert cfg.empty() is False

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            IntegrationsConfigOverride(tavily_api_key="tvly-a", rogue=1)

    def test_digest_request_merges_nested_and_flat_overrides(self) -> None:
        req = DigestRequest(
            topic="quantum computing",
            integrations_config=IntegrationsConfigOverride(
                tavily_api_key="tvly-nested",
                resend_from_email="bot@firm.io",
            ),
            resend_api_key=" re_flat ",
            discord_webhook_url=" https://discord.com/api/webhooks/9/z ",
        )
        assert req.integration_overrides() == {
            "tavily_api_key": "tvly-nested",
            "resend_api_key": "re_flat",
            "resend_from_email": "bot@firm.io",
            "discord_webhook_url": "https://discord.com/api/webhooks/9/z",
        }

    def test_flat_field_wins_over_nested(self) -> None:
        req = DigestRequest(
            topic="quantum computing",
            integrations_config={"tavily_api_key": "tvly-nested"},
            tavily_api_key="tvly-flat",
        )
        assert req.integration_overrides()["tavily_api_key"] == "tvly-flat"

    def test_placeholder_flat_field_does_not_mask_nested(self) -> None:
        req = DigestRequest(
            topic="quantum computing",
            integrations_config={"tavily_api_key": "tvly-nested"},
            tavily_api_key="your_tavily_key",
        )
        assert req.integration_overrides()["tavily_api_key"] == "tvly-nested"

    def test_no_overrides_yields_none_values(self) -> None:
        req = DigestRequest(topic="quantum computing")
        assert req.integration_overrides() == {
            "tavily_api_key": None,
            "resend_api_key": None,
            "resend_from_email": None,
            "discord_webhook_url": None,
        }
