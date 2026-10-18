"""PulseDigest AI — autonomous digest agent pipeline.

Pipeline per request: resolve provider config → live Tavily web context →
noise-filtering LLM synthesis with citations → parallel notifications →
async SQLite persistence → structured `DigestResponse`.

Robustness contract:
- Integration credentials resolve **UI override first**, then `.env`.
- `response.content` is normalized (vision/reasoning models may return a list
  of content blocks or nothing at all) so a valid Markdown report always ships.
- Upstream LLM failures raise `LLMUpstreamError` carrying the HTTP status the
  API should return (401 / 400 / 502), never an opaque unhandled 500.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from config import Settings, sanitize_optional_text, sanitize_secret
from database import Database
from llm_factory import KeyResolution, create_chat_model, describe_api_key
from schemas import DigestRequest, DigestResponse, LLMProviderEnum, NotificationStatus
from tools import search_tavily, send_discord_alert, send_email_resend

logger = logging.getLogger(__name__)

# Sorted, deduplicated representation of search results for the prompt
_CITATION_BUDGET_CHARS = 24000

_BASE_SYSTEM_PROMPT = """You are PulseDigest AI, an autonomous research analyst.
You receive live web search results about a topic. Write an executive digest:
1. Filter noise — ignore ads, clickbait and duplicate content.
2. Cite every factual claim with bracketed source indexes, e.g. [1][3].
3. Use exactly three Markdown level-2 sections, in this order: "Key Takeaways",
   "Deep Dive", "Sources" — translated into the required output language.
4. Key Takeaways: 3-5 crisp bullets. Deep Dive: 2-3 tight paragraphs.
5. Sources: a numbered list of the URLs actually used, e.g. "1. <url>".
Be concise, factual, and neutral. Never invent sources."""

_AUTO_LANGUAGE_RULE = """LANGUAGE RULE (CRITICAL): Detect the language of the user's TOPIC and write the ENTIRE digest — every section heading, bullet point, analysis paragraph and source annotation — in the EXACT SAME LANGUAGE as that topic, unless the topic text itself explicitly asks for another language. Never answer a Persian topic in English. Persian topics must use Persian headings such as "## نکات کلیدی", "## تحلیل عمیق" and "## منابع"; English topics use "## Key Takeaways", "## Deep Dive" and "## Sources"; Arabic topics use Arabic headings, and so on."""

_TARGET_LANGUAGE_RULE = """LANGUAGE RULE (CRITICAL): The user explicitly requested {language} as the output language. Write the ENTIRE digest — every section heading, bullet point, analysis paragraph and source annotation — in {language}, regardless of the language of the topic or of the search results. Translate the three section headings into {language} as well."""

# Free-form names/codes → a stable display name for the prompt.
LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "fa": "Persian (فارسی)",
    "persian": "Persian (فارسی)",
    "farsi": "Persian (فارسی)",
    "en": "English",
    "english": "English",
    "ar": "Arabic (العربية)",
    "arabic": "Arabic (العربية)",
    "de": "German (Deutsch)",
    "german": "German (Deutsch)",
    "fr": "French (Français)",
    "french": "French (Français)",
    "es": "Spanish (Español)",
    "spanish": "Spanish (Español)",
    "ru": "Russian (Русский)",
    "russian": "Russian (Русский)",
    "tr": "Turkish (Türkçe)",
    "turkish": "Turkish (Türkçe)",
    "zh": "Chinese (中文)",
    "chinese": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "japanese": "Japanese (日本語)",
}


def resolve_output_language(value: str | None) -> str | None:
    """Canonical target language name, or `None` for "mirror the topic"."""
    cleaned = sanitize_optional_text(value) or "auto"
    if cleaned.lower() in {"auto", "same", "mirror", "default"}:
        return None
    return LANGUAGE_DISPLAY_NAMES.get(cleaned.lower(), cleaned)


def build_system_prompt(output_language: str | None = "auto") -> str:
    """System prompt with either mirroring or an explicit target language."""
    language = resolve_output_language(output_language)
    rule = (
        _AUTO_LANGUAGE_RULE
        if language is None
        else _TARGET_LANGUAGE_RULE.format(language=language)
    )
    return f"{_BASE_SYSTEM_PROMPT}\n\n{rule}"


# Default prompt mirrors the topic language ("auto").
_SYSTEM_PROMPT_TEMPLATE = build_system_prompt("auto")

_EMPTY_REPORT = (
    "## Key Takeaways\n"
    "- The model returned an empty response for this topic.\n"
    "- No usable Markdown was produced, so this fallback report is shown.\n\n"
    "## Deep Dive\n"
    "The upstream model accepted the request but returned no text content. "
    "This typically happens with vision (`-vl`) or reasoning models that were "
    "called without multimodal input. Pick a text-first model from Live Model "
    "Discovery and run the digest again.\n\n"
    "## Sources\n"
    "No sources were cited because no report was generated."
)

# Estimated USD per 1M tokens (input, output) — conservative published rates
PROVIDER_COST_RATES: dict[str, tuple[float, float]] = {
    LLMProviderEnum.OPENROUTER.value: (0.60, 1.10),
    LLMProviderEnum.OPENAI.value: (2.50, 10.00),
    LLMProviderEnum.ANTHROPIC.value: (3.00, 15.00),
    LLMProviderEnum.GOOGLE.value: (0.35, 1.05),
    LLMProviderEnum.NVIDIA_NIM.value: (0.30, 0.90),
    LLMProviderEnum.DEEPSEEK.value: (0.27, 1.10),
    LLMProviderEnum.GROQ.value: (0.59, 0.79),
    LLMProviderEnum.CUSTOM.value: (1.00, 2.00),
}
DEFAULT_COST_RATE: tuple[float, float] = (1.00, 2.00)


class LLMUpstreamError(RuntimeError):
    """Upstream LLM failure carrying the HTTP status the API should return."""

    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using per-provider table rates."""
    in_rate, out_rate = PROVIDER_COST_RATES.get(provider, DEFAULT_COST_RATE)
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


def _extract_usage(message: object) -> tuple[int, int]:
    """Pull (prompt, completion) token counts from a LangChain message."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or {}
    return int(token_usage.get("prompt_tokens", 0)), int(token_usage.get("completion_tokens", 0))


def _block_to_text(block: object) -> str:
    """Flatten one content block (dict / object / str) into text."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type is not None and block_type not in {"text", "output_text", "text_delta"}:
            return ""
        for key in ("text", "content"):
            value = block.get(key)
            if isinstance(value, str):
                return value
        return ""
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    return ""


def normalize_report_content(content: object) -> str:
    """Coerce any LangChain `content` payload into non-empty Markdown text.

    Handles plain strings, lists of multimodal content blocks (vision models
    such as `-vl`), block objects, `None` and blank output.
    """
    if content is None:
        return _EMPTY_REPORT
    if isinstance(content, str):
        return content.strip() or _EMPTY_REPORT
    if isinstance(content, list):
        parts = [text for text in (_block_to_text(block) for block in content) if text.strip()]
        return "\n\n".join(parts).strip() or _EMPTY_REPORT
    if isinstance(content, dict):
        return _block_to_text(content).strip() or _EMPTY_REPORT
    return str(content).strip() or _EMPTY_REPORT


def _error_detail(
    exc: Exception,
    provider: str,
    model_name: str,
    key_resolution: KeyResolution | None = None,
) -> tuple[str, int]:
    """Map an upstream exception to (actionable detail, HTTP status)."""
    key_source = (
        f" Key in use: {key_resolution.describe()}." if key_resolution else ""
    )
    key_hint = (
        f"{key_source} Open the Integrations & API Keys vault, re-paste the key for "
        f"{provider!r} without quotes or whitespace, and confirm it is active on the "
        "provider dashboard."
    )
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        message = (
            f"Upstream LLM rejected the API key for provider {provider!r} "
            f"(401 Unauthorized).{key_hint} Provider said: {_short(exc)}"
        )
        return message, 401
    if isinstance(exc, RateLimitError):
        message = (
            f"Provider {provider!r} rate-limited this request (429 Too Many Requests). "
            "Wait a few seconds, switch to a different model, or add billing credit. "
            f"Provider said: {_short(exc)}"
        )
        return message, 429
    if isinstance(exc, NotFoundError):
        message = (
            f"Model {model_name!r} was not found for provider {provider!r} (404). "
            "Run 'Discover Live Models', pick a model from the live list, and retry."
        )
        return message, 400
    if isinstance(exc, BadRequestError):
        message = (
            f"Provider {provider!r} rejected the request (400) for model {model_name!r}. "
            "Verify the model id is available to your key and that your account/region is "
            f"allowed to use it. Provider said: {_short(exc)}"
        )
        return message, 400
    if isinstance(exc, (APITimeoutError, httpx.TimeoutException)):
        message = (
            f"Upstream LLM timed out for provider {provider!r} ({type(exc).__name__}). "
            "Retry, or choose a faster model from Live Model Discovery."
        )
        return message, 502
    if isinstance(exc, APIConnectionError):
        message = (
            f"Could not reach provider {provider!r} ({type(exc).__name__}). "
            "Check the base URL / network connection and retry."
        )
        return message, 502
    if isinstance(exc, APIStatusError):
        status = int(getattr(exc, "status_code", 0) or 0)
        if status in (401, 403):
            message = (
                f"Upstream LLM denied access for provider {provider!r} ({status})."
                f"{key_hint} Provider said: {_short(exc)}"
            )
            return message, 401
        if status == 429:
            message = (
                f"Provider {provider!r} rate-limited this request (429). Retry shortly "
                f"or switch models. Provider said: {_short(exc)}"
            )
            return message, 429
        if status == 404:
            message = (
                f"Model {model_name!r} is not available for provider {provider!r} (404). "
                "Run Live Model Discovery and pick an available model."
            )
            return message, 400
        if 400 <= status < 500:
            message = (
                f"Provider {provider!r} rejected the request (HTTP {status}) for model "
                f"{model_name!r}: {_short(exc)}"
            )
            return message, 400
        message = (
            f"Provider {provider!r} failed upstream (HTTP {status}) for model "
            f"{model_name!r}: {_short(exc)}"
        )
        return message, 502
    message = (
        f"LLM invocation failed for provider {provider!r} with model {model_name!r}: "
        f"{type(exc).__name__}: {_short(exc)}"
    )
    return message, 502


def _short(exc: Exception, limit: int = 300) -> str:
    """Compact, single-line exception message for API responses."""
    message = " ".join(str(exc).split())
    return message[:limit] or type(exc).__name__


async def _invoke_model(
    model: object,
    messages: list,
    provider: str,
    model_name: str,
    key_resolution: KeyResolution | None = None,
) -> object:
    """Invoke the chat model, translating upstream failures into 401/400/502."""
    try:
        return await model.ainvoke(messages)  # type: ignore[attr-defined]
    except LLMUpstreamError:
        raise
    except Exception as exc:
        detail, status_code = _error_detail(
            exc, provider, model_name, key_resolution
        )
        logger.warning("Upstream LLM failure (%s/%s): %s", provider, model_name, detail)
        raise LLMUpstreamError(detail, status_code=status_code) from exc


def resolve_integrations(
    request: DigestRequest, settings: Settings
) -> dict[str, str | None]:
    """Resolve integration credentials: cleaned UI override → `.env` fallback."""
    overrides = request.integration_overrides()
    return {
        "tavily_api_key": sanitize_secret(overrides["tavily_api_key"])
        or sanitize_secret(settings.tavily_api_key),
        "resend_api_key": sanitize_secret(overrides["resend_api_key"])
        or sanitize_secret(settings.resend_api_key),
        "resend_from_email": sanitize_optional_text(overrides["resend_from_email"])
        or sanitize_optional_text(settings.resend_from_email),
        "discord_webhook_url": sanitize_optional_text(overrides["discord_webhook_url"])
        or sanitize_optional_text(settings.discord_webhook_url),
    }


def _format_search_context(results: list[dict]) -> str:
    """Render Tavily results as numbered citation blocks for the LLM."""
    lines: list[str] = []
    for i, result in enumerate(results, start=1):
        lines.append(
            f"[{i}] {result.get('title', 'Untitled')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )
        if sum(len(line) for line in lines) > _CITATION_BUDGET_CHARS:
            break
    return "\n\n".join(lines)


async def run_digest_pipeline(
    request: DigestRequest,
    settings: Settings,
    db_path: str,
    http_client: httpx.AsyncClient,
) -> DigestResponse:
    """Execute the full autonomous digest pipeline for one request."""
    started = time.perf_counter()

    # 1. Resolve LLM (per-request override wins over global defaults).
    model = create_chat_model(request.llm_config, settings=settings)
    provider = (
        request.llm_config.provider.value
        if request.llm_config
        else settings.default_llm_provider
    )
    model_name = (
        request.llm_config.model_name
        if request.llm_config
        else settings.default_llm_model
    )

    # 2. Resolve integrations (UI override first, then `.env`).
    integrations = resolve_integrations(request, settings)

    # 3. Live web context.
    search_results = await search_tavily(
        request.topic,
        max_results=5,
        api_key=integrations["tavily_api_key"],
        http_client=http_client,
        settings=settings,
    )
    context = _format_search_context(search_results) or "No live sources retrieved."

    # 4. LLM synthesis with latency + token capture. The key fingerprint is
    #    resolved now so any upstream 401 names the exact key source used.
    key_resolution = describe_api_key(
        provider,
        settings,
        request.llm_config.api_key_override if request.llm_config else None,
    )
    target_language = resolve_output_language(request.output_language)
    messages = [
        SystemMessage(content=build_system_prompt(request.output_language)),
        HumanMessage(
            content=(
                f"TOPIC: {request.topic}\n"
                f"OUTPUT LANGUAGE: {target_language or 'same language as the TOPIC'}\n\n"
                f"LIVE WEB RESULTS:\n{context}"
            )
        ),
    ]
    response_message = await _invoke_model(
        model, messages, provider, model_name, key_resolution
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    prompt_tokens, completion_tokens = _extract_usage(response_message)
    report_markdown = normalize_report_content(getattr(response_message, "content", None))
    estimated_cost_usd = estimate_cost(provider, prompt_tokens, completion_tokens)
    logger.info(
        "Digest synthesized: provider=%s latency_ms=%s tokens=%s+%s cost=%.6f",
        provider, latency_ms, prompt_tokens, completion_tokens, estimated_cost_usd,
    )

    # 5. Parallel notification dispatch (graceful degradation per channel).
    notifications: dict[str, NotificationStatus] = {}
    tasks = []
    if request.recipient_email and request.notify_email:
        tasks.append(("email",
            send_email_resend(
                recipient=str(request.recipient_email),
                subject=f"PulseDigest: {request.topic}",
                markdown_content=report_markdown,
                api_key=integrations["resend_api_key"],
                from_email=integrations["resend_from_email"],
                http_client=http_client,
                settings=settings,
            ),
        ))
    if request.notify_discord:
        tasks.append(("discord",
            send_discord_alert(
                webhook_url=integrations["discord_webhook_url"],
                title=f"PulseDigest: {request.topic}",
                summary=report_markdown,
                cost_usd=estimated_cost_usd,
                http_client=http_client,
                settings=settings,
            ),
        ))
    if tasks:
        outcomes = await asyncio.gather(*(task[1] for task in tasks))
        for (channel, _), sent in zip(tasks, outcomes, strict=True):
            notifications[channel] = NotificationStatus(sent=bool(sent))
    for channel in ("email", "discord"):
        notifications.setdefault(channel, NotificationStatus(sent=False, detail="not requested"))

    # 6. Persist into async SQLite and return the structured response.
    db = Database(db_path=db_path)
    await db.connect()
    try:
        row_id = await db.log_digest(
            topic=request.topic,
            provider=provider,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            email_sent=notifications["email"].sent,
            discord_sent=notifications["discord"].sent,
            summary_report=report_markdown,
        )
    finally:
        await db.close()

    return DigestResponse(
        id=row_id,
        topic=request.topic,
        report_markdown=report_markdown,
        provider=provider,
        model_name=model_name,
        tokens_used=prompt_tokens + completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        notifications=notifications,
        created_at=datetime.now(UTC),
        latency_ms=latency_ms,
    )
