# Backend Schema & Data Models Document

## 1. Database Schema (SQLite - `pulsedigest.db`)
### Table: `digest_logs`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique log ID |
| `created_at` | TEXT | NOT NULL (ISO-8601 UTC) | Timestamp of request |
| `topic` | TEXT | NOT NULL | Researched topic |
| `provider` | TEXT | NOT NULL | e.g., `openrouter`, `openai`, `custom` |
| `model_name` | TEXT | NOT NULL | e.g., `openrouter/auto`, `gpt-5.6` |
| `prompt_tokens` | INTEGER | DEFAULT 0 | Input token count |
| `completion_tokens`| INTEGER | DEFAULT 0 | Output token count |
| `total_tokens` | INTEGER | DEFAULT 0 | Combined tokens |
| `estimated_cost_usd`| REAL | DEFAULT 0.0 | Calculated USD cost |
| `email_sent` | BOOLEAN | DEFAULT 0 | Resend delivery status |
| `discord_sent` | BOOLEAN | DEFAULT 0 | Discord webhook status |
| `summary_report` | TEXT | NOT NULL | Generated Markdown digest |

## 2. Pydantic Schemas (`schemas.py`)
- **`LLMProviderEnum`**: `openrouter`, `openai`, `anthropic`, `google`, `nvidia_nim`, `deepseek`, `groq`, `custom`
- **`LLMConfigOverride`**: `provider`, `model_name`, `custom_base_url` (Optional), `api_key_override` (Optional)
- **`IntegrationsConfigOverride`**: `tavily_api_key`, `resend_api_key`, `resend_from_email`, `discord_webhook_url` — all Optional; secrets are stripped of quotes/whitespace and placeholder values collapse to `None`
- **`DigestRequest`**: `topic` (min_length=3), `output_language` (default `"auto"`, max 40 chars; blank/quoted values normalize to `auto`), `recipient_email` (Optional), `notify_discord` (bool), `notify_email` (bool), `llm_config` (Optional `LLMConfigOverride`), `integrations_config` (Optional), plus flat convenience fields `tavily_api_key`, `resend_api_key`, `resend_from_email`, `discord_webhook_url`
  - `integration_overrides()` merges the nested object with the flat fields (flat wins) and returns a dict of cleaned values
- **`DigestResponse`**: `id`, `topic`, `report_markdown`, `provider`, `model_name`, `tokens_used`, `estimated_cost_usd`, `latency_ms` (Optional), `notifications` (dict), `created_at`
- **`DependencyUpdateStatus`**: `package_name`, `installed_version`, `latest_stable_version`, `update_available` (bool), `release_url`

## 3. Credential Resolution Contract
1. `api_key_override` (clean) → sanitized `.env` key → `ProviderAuthError` (`400` on `/api/v1/digest`). `llm_factory.describe_api_key()` reports the winner as `KeyResolution(api_key, source)` with `source` ∈ `override | env | missing`, plus a masked fingerprint (`sk-or-v…4a2b`).
2. Tavily / Resend / Discord follow the same order: cleaned UI override → `.env` → graceful degradation (`sent: false`, mock search).
3. `.env` is live: `config.get_settings()` compares the file's `(path, mtime_ns, size)` on every call and rebuilds `Settings` when it changed. Process environment variables still outrank the file; `reload_settings()` forces a rebuild.
4. OpenRouter Live Model Discovery verifies the resolved key against `GET https://openrouter.ai/api/v1/auth/key` before querying the public `/models` list (which ignores invalid tokens). A rejected key raises `ProviderAuthError` naming the masked key and its source. Chat models also send `HTTP-Referer: https://github.com/pulsedigest-ai` and `X-Title: PulseDigest AI`.
5. Upstream LLM `401/403` details include `Key in use: UI Override (sk-or-v…4a2b)` (or `Server .env (...)`) so the operator knows exactly which credential failed.
6. `GET /api/v1/providers` returns `env_status` booleans plus masked `env_fingerprints`:
   ```json
   {
     "env_status": {
       "tavily_configured": true,
       "resend_configured": true,
       "discord_configured": false,
       "providers_configured": {"openrouter": true, "openai": false, "custom": false}
     },
     "env_fingerprints": {"openrouter": "sk-or-v…4a2b"}
   }
   ```
   `resend_configured` requires both `RESEND_API_KEY` and `RESEND_FROM_EMAIL`; full secret values are never serialized.
- **`ModelDiscoveryRequest`**: `provider` (`LLMProviderEnum`), `api_key_override` (Optional), `custom_base_url` (Optional) — request for Live Model Discovery.
- **`ModelInfo`**: `id` (provider model ID, e.g., `openrouter/auto`), `name` (human-readable display name), `provider` (`LLMProviderEnum`), `context_length` (Optional int)
- **`ModelDiscoveryResponse`**: `provider` (`LLMProviderEnum`), `models: list[ModelInfo]` — live, non-deprecated models returned by the provider's `/models` endpoint.