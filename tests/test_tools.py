"""Unit tests for async external tools (`tools.py`) — all network mocked."""

from __future__ import annotations

import httpx
import pytest
from conftest import make_settings

from tools import (
    _markdown_to_email_html,
    search_tavily,
    send_discord_alert,
    send_email_resend,
)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestSearchTavily:
    async def test_live_search_success(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("https://api.tavily.com/search")
            assert request.headers["Authorization"] == "Bearer tvly-test"
            return httpx.Response(200, json={"results": [
                {"title": "Result", "url": "https://x.io", "content": "Body", "score": 0.9}
            ]})

        client = mock_client(handler)
        try:
            results = await search_tavily(
                "edge ai", api_key="tvly-test", http_client=client,
                settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert results[0]["url"] == "https://x.io"

    async def test_missing_key_returns_mock_fallback(self) -> None:
        client = mock_client(lambda r: pytest.fail("network must not be touched"))
        try:
            results = await search_tavily(
                "quantum", http_client=client, settings=make_settings(tavily_api_key=None)
            )
        finally:
            await client.aclose()
        assert results and "(Mock) quantum" in results[0]["title"]

    @pytest.mark.parametrize("status", [401, 429, 500])
    async def test_http_error_returns_empty_list(self, status: int) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, text="rejected")

        client = mock_client(handler)
        try:
            results = await search_tavily(
                "topic", api_key="tvly-test", http_client=client,
                settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert results == []

    async def test_network_error_returns_empty_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        client = mock_client(handler)
        try:
            results = await search_tavily(
                "topic", api_key="tvly-test", http_client=client,
                settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert results == []


class TestSendEmailResend:
    async def test_success_returns_true(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = request.read()
            assert b"you@example.com" in body
            return httpx.Response(200, json={"id": "email_123"})

        client = mock_client(handler)
        try:
            sent = await send_email_resend(
                "you@example.com", "Digest", "# Report", api_key="re-test",
                from_email="bot@example.com", http_client=client,
                settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert sent is True

    async def test_missing_key_or_sender_returns_false(self) -> None:
        assert await send_email_resend(
            "a@b.com", "s", "m", settings=make_settings(resend_api_key=None)
        ) is False
        assert await send_email_resend(
            "a@b.com", "s", "m", api_key="re-test",
            settings=make_settings(resend_from_email=None),
        ) is False

    async def test_invalid_key_401_returns_false(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "bad key"})

        client = mock_client(handler)
        try:
            sent = await send_email_resend(
                "a@b.com", "s", "m", api_key="wrong", from_email="f@x.com",
                http_client=client, settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert sent is False

    def test_markdown_to_html_escapes(self) -> None:
        html = _markdown_to_email_html("<script>alert(1)</script>\n\nParagraph", "Sub")
        assert "<script>" not in html
        assert "<b>Sub</b>" not in html  # subject escaped
        assert "Sub" in html


class TestSendDiscordAlert:
    async def test_success_embed_payload(self) -> None:
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(204)

        client = mock_client(handler)
        try:
            sent = await send_discord_alert(
                "https://discord.com/api/webhooks/123/abc",
                "PulseDigest: Topic", "Summary here", cost_usd=0.0123,
                http_client=client, settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert sent is True
        assert b'"embeds"' in captured["body"]
        assert b"$0.012300" in captured["body"]

    async def test_empty_webhook_returns_false_no_network(self) -> None:
        client = mock_client(lambda r: pytest.fail("network must not be touched"))
        try:
            sent = await send_discord_alert(
                None, "Title", "Summary", 0.0,
                http_client=client, settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert sent is False

    async def test_http_failure_returns_false(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Unknown Webhook")

        client = mock_client(handler)
        try:
            sent = await send_discord_alert(
                "https://discord.com/api/webhooks/1/bad",
                "Title", "Summary", 0.0,
                http_client=client, settings=make_settings(),
            )
        finally:
            await client.aclose()
        assert sent is False
