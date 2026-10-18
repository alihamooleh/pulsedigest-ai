# Implementation Plan & Verification Milestones

## Phase 1: Project Scaffolding & Config (`config.py`, `schemas.py`)
- Setup modular folder structure, `.env.example`, `.gitignore`, `requirements.txt`.
- Implement Pydantic schemas and multi-provider settings loader.
- **Verification Gate 1:** Run `pytest tests/test_schemas.py` and `ruff check`.

## Phase 2: Async Database & Universal LLM Factory (`database.py`, `llm_factory.py`)
- Implement `aiosqlite` connection lifecycle and telemetry queries.
- Implement `llm_factory.py` supporting all 8 provider modes (OpenAI-compatible `base_url` routing + native headers).
- **Verification Gate 2:** Run `pytest tests/test_database.py tests/test_llm_factory.py`.

## Phase 3: External Tools, Agent Engine & Self-Updater (`tools.py`, `agent_engine.py`, `updater.py`)
- Implement async Tavily search, Resend email sender, Discord webhook dispatcher (with graceful fallback when API keys are dummy/missing in test mode).
- Implement `updater.py` to check PyPI stable releases and trigger notifications.
- **Verification Gate 3:** Run `pytest tests/test_agent_and_updater.py`.

## Phase 4: FastAPI Server (`main.py`), Web UI (`static/index.html`) & Open-Source Polish
- Implement FastAPI with `lifespan`, `app.state`, CORS, and API routes (`/api/v1/digest`, `/api/v1/stats`, `/api/v1/system/check-updates`, `/api/v1/providers`).
- Build the dark-mode Web Playground UI and comprehensive `README.md`.
- **Verification Gate 4:** Run full test suite (`pytest -v`) and linter (`ruff check .`) until 0 errors.

## Phase 5: Hardening, Integration Vault & Premium Console
- **Key hygiene (`config.py`, `llm_factory.py`):** strip quotes/whitespace from every secret, treat placeholder values (`your_...`, `changeme`, `sk-xxx`, `tvly-xxx`, `re_xxx`, `...example...`) as unset, and enforce `api_key_override` → `.env` precedence in `create_chat_model` and `fetch_available_models`.
- **Integration overrides (`schemas.py`, `agent_engine.py`, `main.py`):** `IntegrationsConfigOverride` + flat `DigestRequest` fields for Tavily/Resend/Discord, resolved override-first in `run_digest_pipeline`; `/api/v1/providers` exposes `env_status` booleans only.
- **Resilience (`agent_engine.py`, `main.py`):** normalize list-of-block/empty `response.content` into valid Markdown; map upstream `401`/`400`/`429`/`502` failures to `{"detail": ...}` via `LLMUpstreamError` instead of a `500`.
- **Premium console (`static/index.html`):** glassmorphism dark theme, localStorage key vault, live model search + quick filters, animated pipeline progress, sanitized Markdown reader with copy/download, telemetry/update panel.
- **Verification Gate 5:** `ruff check . --fix` and `pytest -v` → 100% pass, 0 errors.

## Phase 6: Day/Night Theme, Language Mirroring & Key Transparency
- **Theme (`static/index.html`):** header toggle switches `html.dark`/`html.light`, persists in `localStorage` (`pd.theme`), applied before first paint; full light palette for cards, fields, badges, toggles and the Markdown reader.
- **Language (`schemas.py`, `agent_engine.py`, `static/index.html`):** `DigestRequest.output_language = "auto"`; `build_system_prompt()` mirrors the topic language by default (including translated headings such as `## نکات کلیدی`) or forces an explicit target language; UI selector offers Auto / Persian / English / Arabic / German / French.
- **Key transparency (`config.py`, `llm_factory.py`, `main.py`, `static/index.html`):** live `.env` reload via mtime, per-provider browser key storage, `env_fingerprints` in `/api/v1/providers`, `KeyResolution` source + masked fingerprint in 401 details, and OpenRouter `/auth/key` verification during discovery.
- **Verification Gate 6 (Final):** `ruff check . --fix` + `pytest -v` → 100% pass, 0 errors; then browser check both themes, per-provider key switching, output-language selector and the discover flow.