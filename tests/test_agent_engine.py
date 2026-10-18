"""Unit tests for the autonomous digest pipeline (`agent_engine.py`).

LLM and Tavily are faked; DB uses a tmp aiosqlite file; notification tools are
asserted via graceful-degradation paths or monkeypatched fakes.
"""

from __future__ import annotations

import httpx
import pytest
from conftest import make_settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APITimeoutError, AuthenticationError, BadRequestError, NotFoundError

from agent_engine import (
    PROVIDER_COST_RATES,
    LLMUpstreamError,
    _extract_usage,
    _format_search_context,
    build_system_prompt,
    estimate_cost,
    normalize_report_content,
    resolve_integrations,
    resolve_output_language,
    run_digest_pipeline,
)
from schemas import DigestRequest, LLMConfigOverride

REPORT = (
    "## Key Takeaways\n- Fast progress[1]\n- Costs falling[2]\n\n"
    "## Deep Dive\nOn-device inference improved.[1]\n\n"
    "## Sources\n1. https://one.io\n2. https://two.io\n"
)


class FakeChatModel:
    """Records prompt messages; returns a canned AIMessage with token usage."""

    def __init__(self, messages_seen: list) -> None:
        self.messages_seen = messages_seen

    async def ainvoke(self, messages: list) -> AIMessage:
        self.messages_seen.extend(messages)
        return AIMessage(
            content=REPORT,
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


@pytest.fixture()
def configured_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Patch Tavily + chat model into fakes; keep tools' own graceful degradation."""
    messages_seen: list = []
    monkeypatch.setattr(
        "agent_engine.search_tavily",
        _fake_search,
    )
    _patch_fake_llm(monkeypatch, messages_seen)
    settings = make_settings()  # discord_webhook_url=None, resend key present
    settings.tavily_api_key = "tvly-test"
    return messages_seen, settings


def _patch_fake_llm(monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    class SeenFake(FakeChatModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(messages_seen=seen)

    monkeypatch.setattr(
        "agent_engine.create_chat_model",
        lambda config_override=None, settings=None: SeenFake(),
    )


async def _fake_search(query, max_results=5, api_key=None, http_client=None, settings=None):
    return [
        {"title": "Relevant paper", "url": "https://one.io", "content": "solid facts"},
        {"title": "Second", "url": "https://two.io", "content": "more facts"},
    ]


class TestHelpers:
    def test_estimate_cost_uses_provider_table(self) -> None:
        in_rate, out_rate = PROVIDER_COST_RATES["openai"]
        assert estimate_cost("openai", 1000, 500) == pytest.approx(
            (1000 * in_rate + 500 * out_rate) / 1_000_000
        )
        assert estimate_cost("unknown_provider", 10, 10) > 0.0

    def test_extract_usage_prefers_usage_metadata(self) -> None:
        message = AIMessage(
            content="x",
            usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )
        assert _extract_usage(message) == (3, 4)
        assert _extract_usage(object()) == (0, 0)

    def test_format_search_context_numbers_citations(self) -> None:
        context = _format_search_context(
            [{"title": "T", "url": "https://u.io", "content": "C"}]
        )
        assert "[1]" in context and "https://u.io" in context
        assert _format_search_context([]) == ""


def upstream_error(kind, status: int, message: str = "bad key"):
    """Build a real OpenAI SDK status error for error-mapping tests."""
    request = httpx.Request("POST", "https://api.provider.dev/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return kind(f"Error code: {status} - {message}", response=response, body=None)


class TestOutputLanguagePrompt:
    """Task 2: mirror the topic language by default, or force a target language."""

    def test_auto_mirrors_topic_language_with_persian_headers(self) -> None:
        prompt = build_system_prompt("auto")
        assert "EXACT SAME LANGUAGE" in prompt
        assert "## نکات کلیدی" in prompt
        assert "## تحلیل عمیق" in prompt
        assert "## منابع" in prompt
        assert "LANGUAGE RULE (CRITICAL)" in prompt

    @pytest.mark.parametrize("value", ["", "  ", "AUTO", "same"])
    def test_blank_or_auto_uses_mirroring_rule(self, value: str) -> None:
        assert resolve_output_language(value) is None
        assert build_system_prompt(value) == build_system_prompt("auto")

    @pytest.mark.parametrize(
        ("value", "display"),
        [
            ("Persian", "Persian (فارسی)"),
            ("fa", "Persian (فارسی)"),
            ("EN", "English"),
            ("Arabic", "Arabic (العربية)"),
            ("German", "German (Deutsch)"),
            ("French", "French (Français)"),
        ],
    )
    def test_explicit_language_overrides_topic_language(
        self, value: str, display: str
    ) -> None:
        prompt = build_system_prompt(value)
        assert "explicitly requested" in prompt
        assert display in prompt
        assert "EXACT SAME LANGUAGE" not in prompt

    def test_unknown_language_passes_through(self) -> None:
        assert resolve_output_language("Swahili") == "Swahili"


class TestNormalizeReportContent:
    """Task 3: vision/reasoning models may return lists or nothing at all."""

    def test_plain_string_is_trimmed(self) -> None:
        assert normalize_report_content("  # Digest\n- a  ") == "# Digest\n- a"

    def test_list_of_text_blocks_is_joined(self) -> None:
        content = [
            {"type": "text", "text": "## Key Takeaways\n- first"},
            {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
            {"type": "text", "text": "## Sources\n1. https://one.io"},
        ]
        report = normalize_report_content(content)
        assert report.startswith("## Key Takeaways")
        assert "https://one.io" in report
        assert "image_url" not in report
        assert report.count("\n\n") >= 1

    def test_block_objects_and_strings_flatten(self) -> None:
        class Block:
            text = "object block"

        assert normalize_report_content(["raw", Block()]) == "raw\n\nobject block"

    @pytest.mark.parametrize("content", [None, "", "   ", [], [{"type": "image_url"}], {}])
    def test_empty_content_yields_non_empty_markdown(self, content) -> None:
        report = normalize_report_content(content)
        assert report.strip()
        assert "## Key Takeaways" in report
        assert "## Sources" in report

    def test_unknown_scalar_is_stringified(self) -> None:
        assert normalize_report_content(42) == "42"


class TestIntegrationResolution:
    """Task 2: cleaned UI overrides first, then `.env` fallback."""

    def test_overrides_beat_env_values(self) -> None:
        request = DigestRequest(
            topic="edge inference",
            integrations_config={
                "tavily_api_key": ' "tvly-ui" ',
                "resend_api_key": "re_ui",
                "resend_from_email": " ui@firm.io ",
                "discord_webhook_url": "https://discord.com/api/webhooks/7/ui",
            },
        )
        resolved = resolve_integrations(
            request,
            make_settings(
                tavily_api_key="tvly-env",
                resend_api_key="re_env",
                resend_from_email="env@firm.io",
                discord_webhook_url="https://discord.com/api/webhooks/1/env",
            ),
        )
        assert resolved == {
            "tavily_api_key": "tvly-ui",
            "resend_api_key": "re_ui",
            "resend_from_email": "ui@firm.io",
            "discord_webhook_url": "https://discord.com/api/webhooks/7/ui",
        }

    def test_env_fallback_when_only_placeholders_supplied(self) -> None:
        request = DigestRequest(
            topic="edge inference",
            tavily_api_key="your_tavily_key",
            resend_api_key="re_xxx",
        )
        resolved = resolve_integrations(
            request,
            make_settings(
                tavily_api_key="tvly-env",
                resend_api_key="re_env",
                resend_from_email="env@firm.io",
                discord_webhook_url=None,
            ),
        )
        assert resolved["tavily_api_key"] == "tvly-env"
        assert resolved["resend_api_key"] == "re_env"
        assert resolved["resend_from_email"] == "env@firm.io"
        assert resolved["discord_webhook_url"] is None


class TestRunDigestPipeline:
    async def test_end_to_end_success_and_persistence(
        self, configured_pipeline, tmp_path
    ) -> None:
        messages_seen, settings = configured_pipeline
        request = DigestRequest(
            topic="on-device AI inference",
            llm_config=LLMConfigOverride(provider="openai", model_name="gpt-5.6"),
        )
        response = await run_digest_pipeline(
            request, settings, db_path=str(tmp_path / "test.db"), http_client=None  # type: ignore[arg-type]
        )

        assert response.id >= 1
        assert response.provider == "openai"
        assert response.model_name == "gpt-5.6"
        assert response.report_markdown == REPORT.strip()  # normalized (trailing blank lines dropped)
        assert response.tokens_used == 150
        # Cost uses the openai row of the provider rate table
        in_rate, out_rate = PROVIDER_COST_RATES["openai"]
        assert response.estimated_cost_usd == pytest.approx(
            (100 * in_rate + 50 * out_rate) / 1_000_000
        )
        # Notifications: neither channel requested → both present, unsent
        assert response.notifications["email"].sent is False
        assert response.notifications["discord"].sent is False

        # Prompt engineering assertions
        system = next(m for m in messages_seen if isinstance(m, SystemMessage))
        human = next(m for m in messages_seen if isinstance(m, HumanMessage))
        for required in ("Key Takeaways", "Deep Dive", "Sources", "noise"):
            assert required in system.content
        assert "on-device AI inference" in human.content
        assert "https://one.io" in human.content

        # Persistence via a fresh Database connection
        from database import Database

        db = Database(db_path=str(tmp_path / "test.db"))
        await db.connect()
        try:
            row = await db.get_digest(response.id)
        finally:
            await db.close()
        assert row is not None
        assert row["topic"] == "on-device AI inference"
        assert row["summary_report"] == REPORT.strip()
        assert row["total_tokens"] == 150

    async def test_notification_channels_dispatched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr("agent_engine.search_tavily", _fake_search)
        _patch_fake_llm(monkeypatch, [])
        calls: list[str] = []

        async def fake_email(*args, **kwargs) -> bool:
            calls.append("email")
            return True

        async def fake_discord(*args, **kwargs) -> bool:
            calls.append("discord")
            return True

        monkeypatch.setattr("agent_engine.send_email_resend", fake_email)
        monkeypatch.setattr("agent_engine.send_discord_alert", fake_discord)

        request = DigestRequest(
            topic="rag pipelines",
            recipient_email="team@example.com",
            notify_email=True,
            notify_discord=True,
        )
        response = await run_digest_pipeline(
            request, make_settings(), db_path=str(tmp_path / "n.db"), http_client=None  # type: ignore[arg-type]
        )
        assert calls == ["email", "discord"]
        assert response.notifications["email"].sent is True
        assert response.notifications["discord"].sent is True


        from database import Database
        db = Database(db_path=str(tmp_path / "n.db"))
        await db.connect()
        try:
            row = await db.get_digest(response.id)
        finally:
            await db.close()
        assert row["email_sent"] == 1 and row["discord_sent"] == 1

    async def test_ui_overrides_reach_every_integration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
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

        monkeypatch.setattr("agent_engine.search_tavily", fake_search)
        monkeypatch.setattr("agent_engine.send_email_resend", fake_email)
        monkeypatch.setattr("agent_engine.send_discord_alert", fake_discord)
        _patch_fake_llm(monkeypatch, [])

        request = DigestRequest(
            topic="sovereign AI",
            recipient_email="team@firm.io",
            notify_email=True,
            notify_discord=True,
            integrations_config={
                "tavily_api_key": " tvly-ui ",
                "resend_api_key": "re_ui",
                "resend_from_email": "ui@firm.io",
                "discord_webhook_url": "https://discord.com/api/webhooks/7/ui",
            },
        )
        settings = make_settings(
            tavily_api_key="tvly-env",
            resend_api_key="re_env",
            resend_from_email="env@firm.io",
            discord_webhook_url="https://discord.com/api/webhooks/1/env",
        )
        response = await run_digest_pipeline(
            request, settings, db_path=str(tmp_path / "o.db"), http_client=None  # type: ignore[arg-type]
        )

        assert captured["tavily"] == "tvly-ui"
        assert captured["email"] == ("re_ui", "ui@firm.io")
        assert captured["discord"] == "https://discord.com/api/webhooks/7/ui"
        assert response.notifications["email"].sent is True
        assert response.notifications["discord"].sent is True

    async def test_persian_topic_mirrors_language_in_system_prompt(
        self, configured_pipeline, tmp_path
    ) -> None:
        messages_seen, settings = configured_pipeline
        await run_digest_pipeline(
            DigestRequest(topic="خبر های مربوط به ایجنت های هوش مصنوعی"),
            settings,
            db_path=str(tmp_path / "fa.db"),
            http_client=None,  # type: ignore[arg-type]
        )
        system = next(m for m in messages_seen if isinstance(m, SystemMessage))
        human = next(m for m in messages_seen if isinstance(m, HumanMessage))
        assert "EXACT SAME LANGUAGE" in system.content
        assert "نکات کلیدی" in system.content
        assert "same language as the TOPIC" in human.content

    async def test_explicit_output_language_reaches_prompt(
        self, configured_pipeline, tmp_path
    ) -> None:
        messages_seen, settings = configured_pipeline
        await run_digest_pipeline(
            DigestRequest(topic="AI agents", output_language="Persian"),
            settings,
            db_path=str(tmp_path / "fa-explicit.db"),
            http_client=None,  # type: ignore[arg-type]
        )
        system = next(m for m in messages_seen if isinstance(m, SystemMessage))
        human = next(m for m in messages_seen if isinstance(m, HumanMessage))
        assert "Persian (فارسی)" in system.content
        assert "explicitly requested" in system.content
        assert "OUTPUT LANGUAGE: Persian (فارسی)" in human.content

    async def test_list_content_is_normalized_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        class ListContentModel:
            async def ainvoke(self, messages) -> AIMessage:
                return AIMessage(
                    content=[
                        {"type": "text", "text": "## Key Takeaways\n- vision safe"},
                        {"type": "image_url", "image_url": {"url": "https://img/1.png"}},
                    ],
                    usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                )

        monkeypatch.setattr("agent_engine.search_tavily", _fake_search)
        monkeypatch.setattr("agent_engine.create_chat_model", lambda *a, **kw: ListContentModel())
        response = await run_digest_pipeline(
            DigestRequest(topic="vision models"),
            make_settings(),
            db_path=str(tmp_path / "v.db"),
            http_client=None,  # type: ignore[arg-type]
        )
        assert "vision safe" in response.report_markdown
        assert response.tokens_used == 6
        assert response.latency_ms is not None and response.latency_ms >= 0

    async def test_upstream_auth_error_maps_to_401(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        class BoomModel:
            async def ainvoke(self, messages):
                raise upstream_error(AuthenticationError, 401, "Incorrect API key provided")

        monkeypatch.setattr("agent_engine.search_tavily", _fake_search)
        monkeypatch.setattr("agent_engine.create_chat_model", lambda *a, **kw: BoomModel())
        with pytest.raises(LLMUpstreamError) as info:
            await run_digest_pipeline(
                DigestRequest(
                    topic="auth failure",
                    llm_config=LLMConfigOverride(provider="openai", model_name="gpt-5.6"),
                ),
                make_settings(),
                db_path=str(tmp_path / "a.db"),
                http_client=None,  # type: ignore[arg-type]
            )
        assert info.value.status_code == 401
        assert "API key" in info.value.detail
        assert "Incorrect API key provided" in info.value.detail

    async def test_upstream_401_reports_key_source_and_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        class BoomModel:
            async def ainvoke(self, messages):
                raise upstream_error(AuthenticationError, 401, "No auth credentials found")

        monkeypatch.setattr("agent_engine.search_tavily", _fake_search)
        monkeypatch.setattr("agent_engine.create_chat_model", lambda *a, **kw: BoomModel())

        with pytest.raises(LLMUpstreamError) as info:
            await run_digest_pipeline(
                DigestRequest(
                    topic="kimi k2 agents",
                    llm_config=LLMConfigOverride(
                        provider="nvidia_nim",
                        model_name="moonshotai/kimi-k2",
                        api_key_override="nvapi-ui-secret-4242",
                    ),
                ),
                make_settings(nvidia_nim_api_key="nvapi-env-secret-1111"),
                db_path=str(tmp_path / "keys.db"),
                http_client=None,  # type: ignore[arg-type]
            )
        detail = info.value.detail
        assert info.value.status_code == 401
        assert "UI Override (nvapi-u…4242)" in detail
        assert "nvapi-ui-secret-4242" not in detail

        class EnvBoomModel:
            async def ainvoke(self, messages):
                raise upstream_error(AuthenticationError, 401, "bad env key")

        monkeypatch.setattr(
            "agent_engine.create_chat_model", lambda *a, **kw: EnvBoomModel()
        )
        with pytest.raises(LLMUpstreamError) as env_info:
            await run_digest_pipeline(
                DigestRequest(
                    topic="kimi k2 agents",
                    llm_config=LLMConfigOverride(
                        provider="nvidia_nim", model_name="moonshotai/kimi-k2"
                    ),
                ),
                make_settings(nvidia_nim_api_key="nvapi-env-secret-1111"),
                db_path=str(tmp_path / "keys-env.db"),
                http_client=None,  # type: ignore[arg-type]
            )
        assert "Server .env (nvapi-e…1111)" in env_info.value.detail

    async def test_upstream_404_maps_to_400_and_timeout_to_502(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cases = [
            (NotFoundError, 404, 400, "not found"),
            (BadRequestError, 400, 400, "Data policy"),
        ]
        for kind, upstream_status, expected, needle in cases:
            class BoomModel:
                async def ainvoke(
                    self, messages, _kind=kind, _status=upstream_status, _needle=needle
                ):
                    raise upstream_error(_kind, _status, _needle)

            monkeypatch.setattr("agent_engine.search_tavily", _fake_search)
            monkeypatch.setattr("agent_engine.create_chat_model", lambda *a, **kw: BoomModel())
            with pytest.raises(LLMUpstreamError) as info:
                await run_digest_pipeline(
                    DigestRequest(
                        topic="model mapping",
                        llm_config=LLMConfigOverride(
                            provider="openai", model_name="ghost-model"
                        ),
                    ),
                    make_settings(),
                    db_path=str(tmp_path / "m.db"),
                    http_client=None,  # type: ignore[arg-type]
                )
            assert info.value.status_code == expected

        class TimeoutModel:
            async def ainvoke(self, messages):
                request = httpx.Request("POST", "https://api.provider.dev/v1/chat/completions")
                raise APITimeoutError(request=request)

        monkeypatch.setattr("agent_engine.create_chat_model", lambda *a, **kw: TimeoutModel())
        with pytest.raises(LLMUpstreamError) as info:
            await run_digest_pipeline(
                DigestRequest(topic="timeout path"),
                make_settings(),
                db_path=str(tmp_path / "t.db"),
                http_client=None,  # type: ignore[arg-type]
            )
        assert info.value.status_code == 502
        assert "timed out" in info.value.detail.lower()
