"""Unit tests for the universal LLM factory & Live Model Discovery (`llm_factory.py`).

All discovery tests use `httpx.MockTransport` — no network calls are made.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_openai import ChatOpenAI

import config as config_module
from config import Settings
from llm_factory import (
    ANTHROPIC_BASE_URL,
    OPENAI_COMPATIBLE_BASE_URLS,
    ProviderAuthError,
    create_chat_model,
    describe_api_key,
    fetch_available_models,
    is_openai_compatible,
)
from schemas import LLMConfigOverride, LLMProviderEnum


def make_settings(**overrides) -> Settings:
    """Explicit settings so ambient env vars cannot leak into tests."""
    base = {
        "default_llm_provider": "openrouter",
        "default_llm_model": "openrouter/auto",
        "openrouter_api_key": "sk-or-test",
        "openai_api_key": "sk-oa-test",
        "anthropic_api_key": "sk-ant-test",
        "google_api_key": "google-test",
        "nvidia_nim_api_key": "nvidia-test",
        "deepseek_api_key": "deepseek-test",
        "groq_api_key": "groq-test",
        "custom_llm_api_key": "custom-test",
        "custom_llm_base_url": None,
        "_env_file": None,  # do not read a persisted .env
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class TestCreateChatModelDefaults:
    def test_default_provider_openrouter_auto(self) -> None:
        settings = make_settings()
        model = create_chat_model(None, settings=settings)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "openrouter/auto"
        assert str(model.openai_api_base or "") == OPENAI_COMPATIBLE_BASE_URLS["openrouter"]

    def test_default_wrong_provider_name(self) -> None:
        settings = make_settings(default_llm_provider="flip-device")
        with pytest.raises(ValueError, match="Unsupported|base URL"):
            create_chat_model(None, settings=settings)

    def test_openai_direct_provider(self) -> None:
        model = create_chat_model(
            LLMConfigOverride(provider="openai", model_name="gpt-5.6"),
            settings=make_settings(),
        )
        assert model.model_name == "gpt-5.6"
        assert str(model.openai_api_base) == OPENAI_COMPATIBLE_BASE_URLS["openai"]

    def test_anthropic_supported_via_built_in_compat(self) -> None:
        provider = LLMProviderEnum.ANTHROPIC
        assert not is_openai_compatible(provider)
        model = create_chat_model(
            LLMConfigOverride(provider=provider, model_name="claude-4-sonnet"),
            settings=make_settings(),
        )
        assert str(model.openai_api_base or "").startswith(ANTHROPIC_BASE_URL)


class TestCreateChatModelOverrides:
    def test_all_eight_providers_resolve(self) -> None:
        settings = make_settings(custom_llm_base_url="https://custom.host/v1")
        for provider in LLMProviderEnum:
            model = create_chat_model(
                LLMConfigOverride(
                    provider=provider,
                    model_name="testmodel/1",
                    custom_base_url=(
                        "https://custom.host/v1"
                        if provider == LLMProviderEnum.CUSTOM
                        else None
                    ),
                ),
                settings=settings,
            )
            assert isinstance(model, ChatOpenAI)
            assert model.model_name == "testmodel/1"

    def test_custom_without_base_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="base URL"):
            create_chat_model(
                LLMConfigOverride(provider="custom", model_name="m"),
                settings=make_settings(),
            )

    def test_missing_api_key_rejected(self) -> None:
        with pytest.raises(ProviderAuthError, match="openrouter"):
            create_chat_model(
                LLMConfigOverride(provider="openrouter", model_name="openrouter/auto"),
                settings=make_settings(openrouter_api_key=None),
            )


class TestKeySanitizationAndPrecedence:
    """Task 1: quote stripping, placeholder rejection, override precedence."""

    def test_env_keys_strip_quotes_and_whitespace(self) -> None:
        settings = make_settings(
            openrouter_api_key='  "sk-or-real"  ',
            tavily_api_key="'tvly-real'",
            discord_webhook_url="  https://discord.com/api/webhooks/1/abc  ",
        )
        assert settings.openrouter_api_key == "sk-or-real"
        assert settings.tavily_api_key == "tvly-real"
        assert settings.discord_webhook_url == "https://discord.com/api/webhooks/1/abc"

    @pytest.mark.parametrize(
        "placeholder",
        [
            "your_openai_key",
            "your-openai-key",
            "changeme",
            "change_me",
            "change-me",
            "sk-xxx",
            "tvly-xxx",
            "re_xxx",
            "placeholder-key",
            "an-example-api-key",
            "<insert key>",
        ],
    )
    def test_placeholder_env_values_become_none(self, placeholder: str) -> None:
        settings = make_settings(
            openai_api_key=placeholder,
            tavily_api_key=placeholder,
            resend_api_key=placeholder,
            discord_webhook_url=placeholder,
        )
        assert settings.openai_api_key is None
        assert settings.tavily_api_key is None
        assert settings.resend_api_key is None
        assert settings.discord_webhook_url is None

    def test_blank_env_value_becomes_none(self) -> None:
        settings = make_settings(openai_api_key="   ", tavily_api_key="''")
        assert settings.openai_api_key is None
        assert settings.tavily_api_key is None

    def test_resend_email_is_not_treated_as_secret(self) -> None:
        settings = make_settings(resend_from_email="digest@example.com")
        assert settings.resend_from_email == "digest@example.com"

    def test_ui_override_strictly_wins_over_env(self) -> None:
        model = create_chat_model(
            LLMConfigOverride(
                provider="openai",
                model_name="gpt-5.6",
                api_key_override='  "sk-ui-override"  ',
            ),
            settings=make_settings(openai_api_key="sk-env-value"),
        )
        assert model.openai_api_key is not None
        assert model.openai_api_key.get_secret_value() == "sk-ui-override"

    def test_placeholder_override_falls_back_to_env_key(self) -> None:
        model = create_chat_model(
            LLMConfigOverride(
                provider="openai",
                model_name="gpt-5.6",
                api_key_override="your_openai_key",
            ),
            settings=make_settings(openai_api_key="sk-env-value"),
        )
        assert model.openai_api_key is not None
        assert model.openai_api_key.get_secret_value() == "sk-env-value"

    def test_placeholder_env_key_with_no_override_raises_auth_error(self) -> None:
        with pytest.raises(ProviderAuthError, match="No API key"):
            create_chat_model(
                LLMConfigOverride(provider="openai", model_name="gpt-5.6"),
                settings=make_settings(openai_api_key="your_openai_key"),
            )

    def test_custom_base_url_quotes_stripped(self) -> None:
        model = create_chat_model(
            LLMConfigOverride(
                provider="custom",
                model_name="m",
                custom_base_url=' "https://custom.host/v1/" ',
            ),
            settings=make_settings(),
        )
        assert str(model.openai_api_base).rstrip("/") == "https://custom.host/v1"


# ---------------------------------------------------------------------------
# Live Model Discovery (`fetch_available_models`)
# ---------------------------------------------------------------------------


def discovery_client(handler) -> httpx.AsyncClient:
    """Async client wired to a mock transport (caller must aclose())."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestDiscoverOpenAICompatible:
    async def test_openrouter_live_models_filtered_and_parsed(self) -> None:
        payload = {
            "data": [
                {
                    "id": "openrouter/auto",
                    "display_name": "Auto Router",
                    "context_length": 200000,
                },
                {
                    "id": "vendor/legacy-deprecated-ver",
                    "display_name": "Legacy",
                    "context_length": 8192,
                },
                {
                    "id": "vendor/big",
                    "display_name": "Big",
                    "top_provider": {"context_length": 1_000_000},
                },
            ]
        }

        seen: dict[str, str | None] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen[request.url.path] = request.headers.get("authorization")
            if request.url.path.endswith("/auth/key"):
                return httpx.Response(200, json={"data": {"label": "test-key"}})
            assert request.url == httpx.URL(
                "https://openrouter.ai/api/v1/models"
            )
            assert request.headers["Authorization"] == "Bearer sk-or-test"
            return httpx.Response(200, json=payload)

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "openrouter", http_client=client, settings=make_settings()
            )
        finally:
            await client.aclose()

        assert [m.id for m in models] == ["openrouter/auto", "vendor/big"]
        assert models[0].name == "Auto Router"
        assert models[0].provider == LLMProviderEnum.OPENROUTER
        assert models[0].context_length == 200000
        assert models[1].context_length == 1_000_000
        # Public /models ignores bad tokens, so the key is verified first.
        assert seen["/api/v1/auth/key"] == "Bearer sk-or-test"
        assert seen["/api/v1/models"] == "Bearer sk-or-test"

    async def test_override_key_wins_in_discovery_request(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"data": [{"id": "live-1", "name": "Live 1"}]})

        client = discovery_client(handler)
        try:
            await fetch_available_models(
                "openai",
                api_key_override=' "sk-ui-live" ',
                http_client=client,
                settings=make_settings(openai_api_key="sk-env-live"),
            )
        finally:
            await client.aclose()
        assert seen["auth"] == "Bearer sk-ui-live"

    async def test_placeholder_override_falls_back_to_env_in_discovery(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"data": [{"id": "live-1", "name": "Live 1"}]})

        client = discovery_client(handler)
        try:
            await fetch_available_models(
                "openai",
                api_key_override="your_openai_key",
                http_client=client,
                settings=make_settings(openai_api_key="sk-env-live"),
            )
        finally:
            await client.aclose()
        assert seen["auth"] == "Bearer sk-env-live"

    async def test_deepseek_bearer_header(self) -> None:
        seen_headers = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.update(request.headers)
            return httpx.Response(200, json={"data": [{"id": "deepseek-chat", "name": "DeepSeek Chat"}]})

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "deepseek", http_client=client, settings=make_settings()
            )
        finally:
            await client.aclose()

        assert seen_headers["authorization"] == "Bearer deepseek-test"
        assert [m.id for m in models] == ["deepseek-chat"]
        assert models[0].provider == LLMProviderEnum.DEEPSEEK

    async def test_custom_provider_uses_base_url(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "https://api.custom.dev/v1/models"
            return httpx.Response(200, json={"data": [{"id": "custom-1", "name": "Custom 1"}]})

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "custom",
                custom_base_url="https://api.custom.dev/v1",
                http_client=client,
                settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert [m.id for m in models] == ["custom-1"]

    async def test_groq_native_context_window_key(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer groq-test"
            return httpx.Response(200, json={"data": [{"id": "llama-4", "context_window": 131072}]})

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "groq", http_client=client, settings=make_settings()
            )
        finally:
            await client.aclose()
        assert models[0].context_length == 131072
        assert models[0].name == "llama-4"


class TestKeyResolutionSources:
    """Transparency: which key is in use, masked, and where it came from."""

    def test_ui_override_wins_and_is_masked(self) -> None:
        info = describe_api_key(
            "openrouter",
            make_settings(openrouter_api_key="sk-or-env-1234"),
            '  "sk-or-ui-9876"  ',
        )
        assert info.api_key == "sk-or-ui-9876"
        assert info.source == "override"
        assert info.source_label == "UI Override"
        assert info.fingerprint == "sk-or-u…9876"
        assert "UI Override (sk-or-u…9876)" == info.describe()
        assert "sk-or-ui-9876" not in info.describe()

    def test_env_fallback_reports_server_source(self) -> None:
        info = describe_api_key(
            "openrouter", make_settings(openrouter_api_key="sk-or-env-1234")
        )
        assert info.source == "env"
        assert info.fingerprint == "sk-or-e…1234"
        assert info.describe().startswith("Server .env")

    def test_missing_key_is_described(self) -> None:
        info = describe_api_key("openai", make_settings(openai_api_key=None))
        assert info.api_key is None
        assert info.source == "missing"
        assert "missing key" in info.describe()

    def test_openrouter_chat_model_sends_attribution_headers(self) -> None:
        model = create_chat_model(
            LLMConfigOverride(provider="openrouter", model_name="openrouter/auto"),
            settings=make_settings(),
        )
        headers = dict(getattr(model, "default_headers", None) or {})
        assert headers.get("HTTP-Referer") == "https://github.com/pulsedigest-ai"
        assert headers.get("X-Title") == "PulseDigest AI"


class TestOpenRouterKeyVerification:
    """OpenRouter `/auth/key` check during Live Model Discovery."""

    async def test_invalid_override_key_fails_before_models(self) -> None:
        paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(401, json={"error": {"message": "User not found"}})

        client = discovery_client(handler)
        try:
            with pytest.raises(ProviderAuthError) as info:
                await fetch_available_models(
                    "openrouter",
                    api_key_override="sk-or-v1-deadbeef",
                    http_client=client,
                    settings=make_settings(),
                )
        finally:
            await client.aclose()

        message = str(info.value)
        assert "UI Override (sk-or-v…beef)" in message
        assert "https://openrouter.ai/keys" in message
        # Models endpoint is never queried once the key is known to be dead.
        assert paths == ["/api/v1/auth/key"]

    async def test_env_key_is_named_when_rejected(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"message": "forbidden"}})

        client = discovery_client(handler)
        try:
            with pytest.raises(ProviderAuthError) as info:
                await fetch_available_models(
                    "openrouter",
                    http_client=client,
                    settings=make_settings(openrouter_api_key="sk-or-v1-envkey99"),
                )
        finally:
            await client.aclose()
        assert "Server .env (sk-or-v…ey99)" in str(info.value)

    async def test_auth_endpoint_http_error_is_actionable(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="maintenance")

        client = discovery_client(handler)
        try:
            with pytest.raises(ValueError, match="auth check returned HTTP 503"):
                await fetch_available_models(
                    "openrouter", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()

    async def test_auth_endpoint_network_failure_is_actionable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("tls failure")

        client = discovery_client(handler)
        try:
            with pytest.raises(ValueError, match="Could not reach the OpenRouter auth"):
                await fetch_available_models(
                    "openrouter", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()


class TestEnvLiveReload:
    """Editing `.env` while uvicorn runs must take effect on the next read."""

    def test_get_settings_picks_up_edited_env_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import os
        import time

        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-first\nDEFAULT_LLM_MODEL=first/model\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_module, "ENV_FILE", env_file)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)
        config_module.reset_settings()
        try:
            first = config_module.get_settings()
            assert first.openrouter_api_key == "sk-or-first"
            assert first.default_llm_model == "first/model"

            env_file.write_text(
                "OPENROUTER_API_KEY=sk-or-second\nDEFAULT_LLM_MODEL=second/model\n",
                encoding="utf-8",
            )
            os.utime(env_file, (time.time() + 5, time.time() + 5))
            second = config_module.get_settings()
            assert second is not first
            assert second.openrouter_api_key == "sk-or-second"
            assert second.default_llm_model == "second/model"
        finally:
            config_module.reset_settings()


class TestDiscoverAnthropic:
    async def test_native_headers_and_models_key(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["x-api-key"] = request.headers.get("x-api-key")
            seen["anthropic-version"] = request.headers.get("anthropic-version")
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "claude-4-sonnet", "display_name": "Claude 4 Sonnet"},
                        {"id": "claude-3-deprecated", "display_name": "Old"},
                    ]
                },
            )

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "anthropic", http_client=client, settings=make_settings()
            )
        finally:
            await client.aclose()

        assert seen["url"] == "https://api.anthropic.com/v1/models"
        assert seen["x-api-key"] == "sk-ant-test"
        assert seen["anthropic-version"] == "2023-06-01"
        assert seen["auth"] is None  # native key header, no Bearer
        assert [m.id for m in models] == ["claude-4-sonnet"]


class TestDiscoverErrorPaths:
    async def test_invalid_api_key_401(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        client = discovery_client(handler)
        try:
            with pytest.raises(ProviderAuthError, match="Invalid API key"):
                await fetch_available_models(
                    "openai", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()

    async def test_missing_api_key(self) -> None:
        with pytest.raises(ProviderAuthError, match="No API key"):
            await fetch_available_models(
                "groq", settings=make_settings(groq_api_key=None)
            )

    async def test_http_500_raises_value_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = discovery_client(handler)
        try:
            with pytest.raises(ValueError, match="HTTP 500"):
                await fetch_available_models(
                    "openai", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()

    async def test_error_status_overrides_unknown_key(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [
                {"id": "m1", "status": "deprecated"},
                {"id": "m2", "context_size": 4096},
            ]})

        client = discovery_client(handler)
        try:
            models = await fetch_available_models(
                "nvidia_nim", http_client=client, settings=make_settings()
            )
        finally:
            await client.aclose()
        assert [m.id for m in models] == ["m2"]
        assert models[0].context_length == 4096

    async def test_non_json_payload(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>oops</html>")

        client = discovery_client(handler)
        try:
            with pytest.raises(ValueError, match="non-JSON"):
                await fetch_available_models(
                    "openai", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()

    async def test_network_failure_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure")

        client = discovery_client(handler)
        try:
            with pytest.raises(ValueError, match="Failed to reach"):
                await fetch_available_models(
                    "openai", http_client=client, settings=make_settings()
                )
        finally:
            await client.aclose()

    async def test_owns_and_closes_client_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The no-client path must create and close its own AsyncClient."""
        import llm_factory as lf

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "x", "name": "X"}]})

        closed_flags = []

        class RecordingClient(httpx.AsyncClient):
            def __init__(self, *args, **kwargs) -> None:
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

            async def aclose(self) -> None:
                closed_flags.append(True)
                await super().aclose()

        monkeypatch.setattr(lf.httpx, "AsyncClient", RecordingClient)
        models = await lf.fetch_available_models(
            "openai", settings=make_settings()
        )
        assert [m.id for m in models] == ["x"]
        assert closed_flags == [True]
