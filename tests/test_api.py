"""Integration tests for the FastAPI server (`main.py`).

External calls are monkeypatched; SQLite uses a per-test tmp database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from conftest import make_settings
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from openai import AuthenticationError

import agent_engine
import config as config_module
from llm_factory import ProviderAuthError
from main import app
from schemas import DependencyUpdateStatus, DigestResponse, ModelInfo


class FakeModel:
    async def ainvoke(self, messages) -> AIMessage:
        return AIMessage(
            content="## Key Takeaways\n- ok\n\n## Sources\n1. https://x.io",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    """TestClient with lifespan; DB isolated to a tmp file per test."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api-test.db"))
    config_module.reset_settings()
    with TestClient(app) as test_client:
        yield test_client
    config_module.reset_settings()


class TestStaticAndProviders:
    def test_serves_dark_dashboard(self, client: TestClient) -> None:
        response = client.get("/", headers={"accept": "text/html"})
        assert response.status_code == 200
        assert "PulseDigest" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_providers_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        body = response.json()
        assert set(body["providers"]) == {
            "openrouter", "openai", "anthropic", "google",
            "nvidia_nim", "deepseek", "groq", "custom",
        }
        assert body["default_model"] == "openrouter/auto"

    def test_dashboard_ships_live_search_and_key_vault(self, client: TestClient) -> None:
        html = client.get("/").text
        for needle in (
            "model-search",
            "filter-chips",
            "Discover Live Models",
            "Search 400+ models by name or ID",
            "tavily-api-key",
            "resend-api-key",
            "resend-from-email",
            "discord-webhook-url",
            "Clear saved keys",
            "localStorage",
            "dir=\"auto\"",
            # Day / night theme toggle
            "theme-toggle",
            "theme-label",
            "html.light",
            # Output language selector
            "output-language",
            "Auto (Same as Topic Language)",
            "Persian (فارسی)",
            "Arabic (العربية)",
            # Per-provider key vault + transparency badge
            "providerKeys",
            "key-source",
            "Use server .env instead",
            "Active key: UI Override",
        ):
            assert needle in html, f"dashboard missing {needle!r}"

    def test_light_theme_paints_fields_without_resetting_select_chevron(
        self, client: TestClient
    ) -> None:
        """Regression: a `background` shorthand in html.light tiled the chevron arrow."""
        html = client.get("/").text

        # Light-mode fields must set only background-color, keeping the chevron longhands intact.
        assert "html.light .field { background-color:" in html
        assert "html.light .field { background:" not in html
        assert "html.light .field:focus {\n      background-color:" in html
        assert "html.light select.field option" in html

        # Both dark and light select rules restate every chevron longhand.
        chevron_rule = html.split("select.field, html.light select.field {", 1)[1].split("}", 1)[0]
        for longhand in (
            "background-repeat: no-repeat",
            "background-position: right .7rem center",
            "background-size: 14px 14px",
        ):
            assert longhand in chevron_rule
        light_select = html.split("html.light select.field {", 1)[1].split("}", 1)[0]
        for longhand in (
            "background-repeat: no-repeat",
            "background-position: right .7rem center",
            "background-size: 14px 14px",
        ):
            assert longhand in light_select

        # Softer muted-slate day palette (no pure-white canvas/cards).
        assert "html.light body { background-color: #DCE3EC;" in html
        assert "html.light .glass-soft { background: rgba(241,245,249," in html
        assert "#FFFFFF" not in light_select

        # The search box has its own SVG icon, so the placeholder carries no emoji.
        assert 'placeholder="Search 400+ models by name or ID…"' in html
        assert "🔍 Search 400+" not in html

    def test_providers_env_status_booleans_without_secrets(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = make_settings(
            openai_api_key="sk-super-secret-openai",
            anthropic_api_key=None,
            tavily_api_key="tvly-super-secret",
            resend_api_key="re_super_secret",
            resend_from_email="bot@firm.io",
            discord_webhook_url=None,
        )
        monkeypatch.setattr("main.get_settings", lambda: settings)

        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        body = response.json()
        env_status = body["env_status"]

        assert env_status["tavily_configured"] is True
        assert env_status["resend_configured"] is True
        assert env_status["discord_configured"] is False
        assert set(env_status["providers_configured"]) == {
            "openrouter", "openai", "anthropic", "google",
            "nvidia_nim", "deepseek", "groq", "custom",
        }
        assert env_status["providers_configured"]["openai"] is True
        assert env_status["providers_configured"]["anthropic"] is False
        assert all(isinstance(v, bool) for v in env_status["providers_configured"].values())

        # Masked fingerprints reveal the key source without leaking the secret.
        fingerprints = body["env_fingerprints"]
        assert fingerprints["openai"] == "sk-supe…enai"
        assert "anthropic" not in fingerprints
        assert "openrouter" in fingerprints

        # Never leak the actual secret values.
        serialized = response.text
        for secret in ("sk-super-secret-openai", "tvly-super-secret", "re_super_secret"):
            assert secret not in serialized


class TestDigestEndpoint:
    def test_digest_runs_pipeline_and_stats_update(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_search(*args, **kwargs):
            return [{"title": "S", "url": "https://s.io", "content": "facts"}]

        monkeypatch.setattr(agent_engine, "search_tavily", fake_search)
        monkeypatch.setattr(agent_engine, "create_chat_model", lambda *a, **kw: FakeModel())

        response = client.post(
            "/api/v1/digest",
            json={
                "topic": "vector databases 2026",
                "notify_discord": False,
                "llm_config": {"provider": "openrouter", "model_name": "openrouter/auto"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["topic"] == "vector databases 2026"
        assert body["provider"] == "openrouter"
        assert body["model_name"] == "openrouter/auto"
        assert body["tokens_used"] == 15
        assert body["notifications"]["email"]["sent"] is False

        stats = client.get("/api/v1/stats").json()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 15

    def test_digest_starts_invalid_topic_returns_422(self, client: TestClient) -> None:
        assert client.post("/api/v1/digest", json={"topic": "ab"}).status_code == 422

    def test_digest_pipeline_error_maps_to_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(*args, **kwargs):
            raise RuntimeError("LLM exploded")

        monkeypatch.setattr("main.run_digest_pipeline", boom)
        response = client.post("/api/v1/digest", json={"topic": "valid topic here"})
        assert response.status_code == 500

    def test_upstream_401_returns_clean_json_detail(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_search(*args, **kwargs):
            return []

        class UnauthorizedModel:
            async def ainvoke(self, messages):
                request = httpx.Request("POST", "https://api.provider.dev/v1/chat/completions")
                response = httpx.Response(
                    401,
                    request=request,
                    json={"error": {"message": "Incorrect API key provided"}},
                )
                raise AuthenticationError(
                    "Error code: 401 - Incorrect API key provided",
                    response=response,
                    body=None,
                )

        monkeypatch.setattr(agent_engine, "search_tavily", fake_search)
        monkeypatch.setattr(agent_engine, "create_chat_model", lambda *a, **kw: UnauthorizedModel())

        response = client.post(
            "/api/v1/digest",
            json={
                "topic": "upstream auth failure",
                "llm_config": {"provider": "openai", "model_name": "gpt-5.6"},
            },
        )
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert "API key" in detail
        assert "Incorrect API key provided" in detail

    def test_list_content_is_normalized_in_response(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_search(*args, **kwargs):
            return [{"title": "T", "url": "https://t.io", "content": "C"}]

        class VisionModel:
            async def ainvoke(self, messages) -> AIMessage:
                return AIMessage(
                    content=[
                        {"type": "text", "text": "## Key Takeaways\n- normalized list block"},
                        {"type": "image_url", "image_url": {"url": "https://img/1.png"}},
                    ],
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                )

        monkeypatch.setattr(agent_engine, "search_tavily", fake_search)
        monkeypatch.setattr(agent_engine, "create_chat_model", lambda *a, **kw: VisionModel())

        response = client.post("/api/v1/digest", json={"topic": "vision model list payload"})
        assert response.status_code == 200
        body = response.json()
        assert "normalized list block" in body["report_markdown"]
        assert body["tokens_used"] == 10
        assert body["latency_ms"] is not None

    def test_output_language_flows_to_pipeline(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        async def fake_pipeline(request, settings, db_path, http_client):
            captured["language"] = request.output_language
            return DigestResponse(
                id=1,
                topic=request.topic,
                report_markdown="# ok",
                provider="openai",
                model_name="gpt-5.6",
                created_at=datetime.now(UTC),
            )

        monkeypatch.setattr("main.run_digest_pipeline", fake_pipeline)
        response = client.post(
            "/api/v1/digest",
            json={"topic": "ai agents", "output_language": "Persian"},
        )
        assert response.status_code == 200
        assert captured["language"] == "Persian"

        response = client.post("/api/v1/digest", json={"topic": "ai agents"})
        assert response.status_code == 200
        assert captured["language"] == "auto"

    def test_integration_overrides_flow_through_api(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        async def fake_search(query, max_results=5, api_key=None, http_client=None, settings=None):
            captured["tavily"] = api_key
            return [{"title": "T", "url": "https://t.io", "content": "C"}]

        async def fake_email(**kwargs) -> bool:
            captured["email"] = (kwargs.get("api_key"), kwargs.get("from_email"))
            return True

        async def fake_discord(**kwargs) -> bool:
            captured["discord"] = kwargs.get("webhook_url")
            return True

        class Model:
            async def ainvoke(self, messages) -> AIMessage:
                return AIMessage(
                    content="## Key Takeaways\n- ok",
                    usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                )

        monkeypatch.setattr(agent_engine, "search_tavily", fake_search)
        monkeypatch.setattr(agent_engine, "send_email_resend", fake_email)
        monkeypatch.setattr(agent_engine, "send_discord_alert", fake_discord)
        monkeypatch.setattr(agent_engine, "create_chat_model", lambda *a, **kw: Model())

        response = client.post(
            "/api/v1/digest",
            json={
                "topic": "override plumbing",
                "recipient_email": "team@firm.io",
                "notify_email": True,
                "notify_discord": True,
                "integrations_config": {
                    "tavily_api_key": " tvly-ui ",
                    "resend_api_key": "re_ui",
                    "resend_from_email": "ui@firm.io",
                    "discord_webhook_url": "https://discord.com/api/webhooks/3/ui",
                },
            },
        )
        assert response.status_code == 200
        assert captured["tavily"] == "tvly-ui"
        assert captured["email"] == ("re_ui", "ui@firm.io")
        assert captured["discord"] == "https://discord.com/api/webhooks/3/ui"
        assert response.json()["notifications"]["email"]["sent"] is True
        assert response.json()["notifications"]["discord"]["sent"] is True


class TestModelDiscoveryEndpoint:
    def test_discover_success(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fetch(provider, api_key_override=None, custom_base_url=None, **kwargs):
            assert provider.value == "openai"
            assert api_key_override == "sk-live"
            return [
                ModelInfo(id="gpt-5.6", name="GPT 5.6", provider=provider),
                ModelInfo(id="gpt-6", name="GPT 6", provider=provider, context_length=1000000),
            ]

        monkeypatch.setattr("main.fetch_available_models", fake_fetch)
        response = client.post(
            "/api/v1/models/discover",
            json={"provider": "openai", "api_key_override": "sk-live"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "openai"
        assert [m["id"] for m in body["models"]] == ["gpt-5.6", "gpt-6"]

    def test_discover_invalid_provider_403(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def deny(*args, api_key_override=None, **kwargs):
            if api_key_override:
                raise ProviderAuthError("Invalid API key for provider 'openai' (401).")
            raise ValueError("custom requires a base URL")

        monkeypatch.setattr("main.fetch_available_models", deny)
        response = client.post(
            "/api/v1/models/discover",
            json={"provider": "openai", "api_key_override": "wrong"},
        )
        assert response.status_code == 403
        response = client.post(
            "/api/v1/models/discover",
            json={"provider": "custom"},  # custom without base URL → 400
        )
        assert response.status_code == 400

    def test_discover_bad_base_url_rejected_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/models/discover",
            json={"provider": "custom", "custom_base_url": "nope"},
        )
        assert response.status_code == 422


class TestStatsAndUpdates:
    def test_stats_empty_database(self, client: TestClient) -> None:
        stats = client.get("/api/v1/stats").json()
        assert stats["total_requests"] == 0
        assert stats["total_cost_usd"] == 0.0
        assert stats["by_provider"] == []

    def test_check_updates_endpoint(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_updates(http_client=None, **kwargs):
            return [
                DependencyUpdateStatus(
                    package_name="fastapi",
                    installed_version="0.143.2",
                    latest_stable_version="0.144.0",
                    update_available=True,
                    release_url="https://pypi.org/project/fastapi/",
                )
            ]

        monkeypatch.setattr("main.check_dependency_updates", fake_updates)
        response = client.get("/api/v1/system/check-updates")
        assert response.status_code == 200
        body = response.json()
        assert body[0]["update_available"] is True
        assert body[0]["package_name"] == "fastapi"
