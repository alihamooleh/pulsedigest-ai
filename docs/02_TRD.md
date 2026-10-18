# Technical Requirements Document (TRD)

## 1. Technical Goals
- **100% Cloud-Native & Zero-GPU Requirement:** Runs anywhere with Python 3.13+ and API keys.
- **Strictly Asynchronous:** Non-blocking I/O across FastAPI endpoints, database queries, and HTTP calls.
- **Self-Testing & Self-Healing:** 100% automated validation via `pytest`, `pytest-asyncio`, and `ruff`.

## 2. Technology Stack
- **Language & Runtime:** Python 3.13+, `venv`
- **Backend Framework:** FastAPI + Uvicorn (Strictly using `lifespan` with `@asynccontextmanager` and `app.state`)
- **Database:** Async SQLite (`aiosqlite`) for zero-config open-source portability
- **AI & Agent Framework:** LangChain Core / LangChain OpenAI / Universal `base_url` Adapter + MCP Doc Fetcher
- **Supported Cloud LLM Providers:**
  - OpenRouter (`https://openrouter.ai/api/v1`)
  - OpenAI (`https://api.openai.com/v1`)
  - Anthropic (`https://api.anthropic.com/v1`)
  - Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai/`)
  - NVIDIA NIM (`https://integrate.api.nvidia.com/v1`)
  - DeepSeek (`https://api.deepseek.com/v1`)
  - Groq (`https://api.groq.com/openai/v1`)
  - Custom Provider (`CUSTOM_LLM_BASE_URL` + arbitrary `model_name`)
- **External Integrations:** Tavily Search API (`httpx` async), Resend Email API, Discord Webhooks
- **Frontend UI:** Lightweight Vanilla JS + Tailwind CSS (CDN) served natively by FastAPI static/template route
- **Quality & Testing Suite:** `pytest`, `pytest-asyncio`, `httpx` (TestClient), `ruff`

## 3. Live Model Discovery
- The backend exposes a `fetch_available_models` capability in `llm_factory.py` (and a `POST /api/v1/models/discover` route):
  - For OpenAI-compatible providers (`openrouter`, `openai`, `google`, `nvidia_nim`, `deepseek`, `groq`, `custom`): async `GET {base_url}/models` with `Authorization: Bearer <key>` via `httpx.AsyncClient`.
  - For Anthropic: `GET https://api.anthropic.com/v1/models` with `x-api-key` + `anthropic-version` headers.
  - Returns `ModelInfo` entries (`id`, `name`, `provider`, `context_length`) for non-deprecated models so the UI model selector is always live and current.
- **Default model fallback:** `openrouter/auto` (auto-routing); never hardcode deprecated model identifiers such as `openrouter/owl-alpha`.