"""PulseDigest AI — async external integrations (Tavily, Resend, Discord).

All network I/O is strictly non-blocking via `httpx.AsyncClient`. Every
integration degrades gracefully: missing API keys produce logged warnings and
safe fallbacks (`sent: False`, mock search results) — never an unhandled crash.
"""

from __future__ import annotations

import html
import logging

import httpx

from config import Settings, get_settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
RESEND_EMAIL_URL = "https://api.resend.com/emails"

_DISCORD_EMBED_COLOR = 0x10B981  # emerald accent


def _own_client(
    http_client: httpx.AsyncClient | None, timeout: float
) -> tuple[httpx.AsyncClient, bool]:
    """Return the client to use plus whether this function owns (must close) it."""
    if http_client is not None:
        return http_client, False
    return httpx.AsyncClient(timeout=timeout), True


async def search_tavily(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Live web search via Tavily; safe mock fallback without an API key.

    Returns a list of result dicts (`title`, `url`, `content`, `score`).
    Missing/empty key or network/HTTP failure never raises: a warning is
    logged and a fallback (mock page for dev/test, or `[]` on wire errors)
    is returned.
    """
    settings = settings or get_settings()
    key = api_key or settings.tavily_api_key or ""
    if not key.strip():
        logger.warning("TAVILY_API_KEY not set — returning mock search results.")
        return [
            {
                "title": f"(Mock) {query}",
                "url": "https://example.com/mock-result",
                "content": (
                    f"Development fallback for topic {query!r}: no live Tavily "
                    "search was performed because no API key is configured. "
                    "This stub keeps the digest pipeline testable end-to-end."
                ),
                "score": 0.5,
            }
        ]

    client, owned = _own_client(http_client, settings.http_timeout_seconds)
    try:
        response = await client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
        )
        if response.status_code >= 400:
            logger.warning(
                "Tavily search failed (HTTP %s): %s", response.status_code, response.text[:200]
            )
            return []
        payload = response.json()
        results = payload.get("results", [])
        return results if isinstance(results, list) else []
    except httpx.HTTPError as exc:
        logger.warning("Tavily search unreachable: %s", exc)
        return []
    finally:
        if owned:
            await client.aclose()


def _markdown_to_email_html(markdown_content: str, subject: str) -> str:
    """Tiny dependency-free Markdown-to-HTML conversion for Digest emails."""
    paragraphs = [html.escape(p) for p in markdown_content.split("\n\n") if p.strip()]
    body = "</p><p>".join(paragraphs)
    return (
        "<html><body>"
        f"<h2>{html.escape(subject)}</h2>"
        f"<p>{body}</p>"
        "<p style='color:#9ca3af;font-size:12px'>"
        "Sent by <strong>PulseDigest AI</strong> — open-source autonomous digests.</p>"
        "</body></html>"
    )


async def send_email_resend(
    recipient: str,
    subject: str,
    markdown_content: str,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    from_email: str | None = None,
    settings: Settings | None = None,
) -> bool:
    """Send an HTML digest email through the Resend API.

    Returns `True` on accepted delivery, `False` on missing/invalid config
    or transport failure (with a logged warning — graceful degradation).
    """
    settings = settings or get_settings()
    key = api_key or settings.resend_api_key or ""
    sender = from_email or settings.resend_from_email or ""
    if not key.strip() or not sender:
        logger.warning(
            "RESEND_API_KEY/RESEND_FROM_EMAIL not set — skipping email to %r.", recipient
        )
        return False

    client, owned = _own_client(http_client, settings.http_timeout_seconds)
    try:
        response = await client.post(
            RESEND_EMAIL_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "from": f"PulseDigest AI <{sender}>",
                "to": [recipient],
                "subject": subject,
                "html": _markdown_to_email_html(markdown_content, subject),
            },
        )
        if 200 <= response.status_code < 300:
            logger.info("Resend email delivered to %r.", recipient)
            return True
        logger.warning(
            "Resend rejected email (HTTP %s): %s", response.status_code, response.text[:200]
        )
        return False
    except httpx.HTTPError as exc:
        logger.warning("Resend API unreachable: %s", exc)
        return False
    finally:
        if owned:
            await client.aclose()


async def send_discord_alert(
    webhook_url: str | None,
    title: str,
    summary: str,
    cost_usd: float,
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> bool:
    """Post a rich Embed card to a Discord webhook.

    Returns `True` on success; `False` when the webhook URL is empty/invalid
    or delivery fails (logged warning — graceful degradation).
    """
    settings = settings or get_settings()
    if not webhook_url or not str(webhook_url).strip():
        logger.info("No Discord webhook URL configured — skipping alert.")
        return False

    client, owned = _own_client(http_client, settings.http_timeout_seconds)
    try:
        response = await client.post(
            str(webhook_url).strip(),
            json={
                "embeds": [
                    {
                        "title": title[:256],
                        "description": summary[:4000],
                        "color": _DISCORD_EMBED_COLOR,
                        "fields": [
                            {
                                "name": "Estimated Cost",
                                "value": f"${cost_usd:.6f}",
                                "inline": True,
                            }
                        ],
                        "footer": {"text": "PulseDigest AI • autonomous digest"},
                    }
                ]
            },
        )
        if 200 <= response.status_code < 300:
            return True
        logger.warning(
            "Discord webhook rejected alert (HTTP %s): %s",
            response.status_code,
            response.text[:200],
        )
        return False
    except httpx.HTTPError as exc:
        logger.warning("Discord webhook unreachable: %s", exc)
        return False
    finally:
        if owned:
            await client.aclose()
