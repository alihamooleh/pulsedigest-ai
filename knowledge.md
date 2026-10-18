# Project Knowledge & Strict Agent Rules (Freebuff / Coding Agent)

## 1. Architectural Laws
- **Python Version:** Python 3.13+
- **FastAPI Lifespan:** ALWAYS use `@asynccontextmanager` with `lifespan` in `main.py` and store shared resources (`db_path`, `http_client`) in `app.state`. NEVER use deprecated `@app.on_event("startup")`.
- **Pydantic v2:** ALWAYS use Pydantic v2 syntax (`model_config = ConfigDict(...)`, `.model_dump()`, `@field_validator`). NEVER use deprecated Pydantic v1 `.dict()` or `class Config:`.
- **Universal Cloud LLMs Only:** NEVER install or use local models (`ollama`, `torch`, `transformers`). Route all LLMs through `llm_factory.py` using cloud providers (`openrouter`, `openai`, `anthropic`, `google`, `nvidia_nim`, `deepseek`, `groq`, `custom`).
- **Default LLM Model:** The global default fallback model is `openrouter/auto` (auto-routing). NEVER hardcode deprecated model identifiers (e.g., `openrouter/owl-alpha`) anywhere in code, config, or docs.
- **Live Model Discovery:** `llm_factory.py` MUST expose `async def fetch_available_models(provider, api_key_override=None, custom_base_url=None, http_client=None) -> list[ModelInfo]` that queries the provider's live `/models` endpoint through `httpx.AsyncClient` (OpenAI-compatible `GET {base_url}/models` with `Authorization: Bearer`; Anthropic uses `GET https://api.anthropic.com/v1/models` with `x-api-key` + `anthropic-version` headers) and returns only currently-available, non-deprecated models so users can always select from an up-to-date list.
- **Graceful Degradation:** If an external API key (like `RESEND_API_KEY` or `DISCORD_WEBHOOK_URL`) is empty or not set, do not crash; log a warning and return `sent: False` in the response metadata. The same applies to Live Model Discovery: an invalid API key or unreachable endpoint raises a controlled `ValueError` surfaced as a friendly error, never a crash.

## 2. Mandatory Test-Driven Self-Healing Loop (CRITICAL)
Before finishing ANY task or returning control to the user, you MUST execute this loop in the terminal:
1. Run `ruff check . --fix` to enforce clean, PEP8-compliant, well-structured code.
2. Run `pytest -v` to execute all unit and integration tests.
3. If ANY test or lint check fails, read the traceback carefully, fix the bug in the source code, and re-run `ruff` and `pytest` until **100% of tests pass with 0 warnings/errors**.
4. Never leave broken imports, unhandled async coroutines, or unclosed `httpx.AsyncClient` sessions.

## 3. Up-to-Date Documentation & Self-Update Rule
- Whenever checking or upgrading packages (`langchain-openai`, `langchain-core`, `fastapi`, `pydantic`), query live PyPI JSON metadata (`https://pypi.org/pypi/<pkg>/json`) to verify the latest stable version and ensure compatibility.