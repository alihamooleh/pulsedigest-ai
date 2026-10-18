# Product Requirements Document (PRD) - PulseDigest AI

## 1. Overview & Product Goal
**PulseDigest AI** is an open-source, cloud-native autonomous web research, digest generation, and self-updating microservice. It searches live web sources on any user-defined topic, filters noise, synthesizes executive reports using any major Cloud LLM provider, dispatches alerts via Email (Resend) and Discord, tracks token costs asynchronously, and monitors its own Python dependencies for stable updates.

## 2. Feature Matrix & Expected Behavior
| Feature ID | Feature Name | Description | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **F-01** | **Universal Cloud LLM Router** | Multi-provider LLM switching via config or per-request payload. | Supports `openrouter`, `openai`, `anthropic`, `google`, `nvidia_nim`, `deepseek`, `groq`, and `custom` (`base_url` + dynamic `model_name` e.g., `gpt-5.6`, `gpt-6`). |
| **F-02** | **Autonomous Web Digest Pipeline** | Live web search via Tavily API + multi-step LLM synthesis. | Fetches top web results, filters irrelevant links, and outputs structured Markdown reports with citations. |
| **F-03** | **Multi-Channel Dispatch** | Automated delivery via Resend API (Email) and Discord Webhook. | Sends formatted HTML/Markdown digest to target email and summary embed to Discord asynchronously without crashing if one channel is unconfigured. |
| **F-04** | **Async Cost & Token Telemetry** | Real-time usage logging in async SQLite (`aiosqlite`). | Calculates and persists prompt/completion tokens, latency (ms), provider, model name, and estimated cost per request. |
| **F-05** | **Self-Updating Dependency Watchdog** | Checks PyPI stable releases & official docs for core packages (`langchain`, `fastapi`, `pydantic`). | Alerts via Discord/Email when stable updates exist and generates a migration report / auto-update trigger. |
| **F-06** | **Interactive Web Playground UI** | Clean, single-page control panel served at `/`. | Allows users to test digests, switch LLM providers/models dynamically, and view live telemetry stats. |
| **F-07** | **Live Model Discovery (`fetch_available_models`)** | When a user provides or configures a provider (`openrouter`, `openai`, `anthropic`, `google`, `nvidia_nim`, `deepseek`, `groq`, or `custom` with `custom_base_url`) and an API key, the backend dynamically queries the provider's `/models` endpoint (handling OpenAI-compatible `/v1/models` as well as Anthropic's `x-api-key` headers) to fetch and return the live list of currently available, non-deprecated models. | Users always select from an up-to-date, provider-verified model list instead of hardcoding model names (the default fallback is `openrouter/auto`, which auto-routes to the best available model). |