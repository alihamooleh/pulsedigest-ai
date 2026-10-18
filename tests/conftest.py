"""Shared pytest helpers for PulseDigest AI tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import schemas` etc. when pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_settings(**overrides):
    """Explicit Settings so ambient env vars cannot leak into tests."""
    import config as config_module
    from config import Settings

    base = {
        "default_llm_provider": "openrouter",
        "default_llm_model": "openrouter/auto",
        "openrouter_api_key": "sk-or-test",
        "openai_api_key": "sk-oa-test",
        "anthropic_api_key": "sk-ant-test",
        "google_api_key": "google-test",
        "nvidia_nim_api_key": "nvidia-test",
        "deepseek_api_key": "deepseek-test",
        "groq_api_key": "groq-test",
        "custom_llm_api_key": "custom-test",
        "custom_llm_base_url": None,
        "tavily_api_key": "tvly-test",
        "resend_api_key": "re-test",
        "resend_from_email": "digest@example.com",
        "discord_webhook_url": None,
        "_env_file": None,  # never read a persisted .env
    }
    base.update(overrides)
    config_module.reset_settings()
    return Settings(**base)
