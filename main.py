"""PulseDigest AI — FastAPI server (stdlib lifespan, CORS, static dashboard)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent_engine import LLMUpstreamError, run_digest_pipeline
from config import get_settings
from database import Database
from llm_factory import ProviderAuthError, fetch_available_models
from schemas import (
    DependencyUpdateStatus,
    DigestRequest,
    DigestResponse,
    LLMProviderEnum,
    ModelDiscoveryRequest,
    ModelDiscoveryResponse,
)
from updater import check_dependency_updates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shared resources: async HTTP client + initialized SQLite database."""
    settings = get_settings()
    app.state.http_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
    app.state.db = Database(db_path=settings.db_path)
    await app.state.db.connect()
    logger.info("PulseDigest AI started (provider=%s model=%s db=%s)",
                settings.default_llm_provider, settings.default_llm_model, settings.db_path)
    yield
    await app.state.db.close()
    await app.state.http_client.aclose()
    app.state.http_client = None
    logger.info("PulseDigest AI stopped — resources released.")


app = FastAPI(
    title="PulseDigest AI",
    version="1.0.0",
    description="Autonomous web research → executive digest → dispatch microservice.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the single-page day/night web console."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.post("/api/v1/digest", response_model=DigestResponse)
async def create_digest(payload: DigestRequest, request: Request) -> DigestResponse:
    """Run the full autonomous digest pipeline for one topic."""
    settings = get_settings()
    try:
        return await run_digest_pipeline(
            request=payload,
            settings=settings,
            db_path=settings.db_path,
            http_client=request.app.state.http_client,
        )
    except LLMUpstreamError as exc:
        # Structured, actionable upstream failure (401 / 400 / 429 / 502).
        logger.warning("Digest upstream error (%s): %s", exc.status_code, exc.detail)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ProviderAuthError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{exc} Open the Integrations & API Keys vault and paste the key for "
                "the selected provider (quotes and spaces are stripped automatically)."
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Digest pipeline failed: {exc}") from exc


@app.post("/api/v1/models/discover", response_model=ModelDiscoveryResponse)
async def discover_models(payload: ModelDiscoveryRequest, request: Request) -> ModelDiscoveryResponse:
    """Live Model Discovery — fetch the provider's current non-deprecated models."""
    try:
        models = await fetch_available_models(
            provider=payload.provider,
            api_key_override=payload.api_key_override,
            custom_base_url=payload.custom_base_url,
            http_client=request.app.state.http_client,
        )
    except ProviderAuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelDiscoveryResponse(provider=payload.provider, models=models)


@app.get("/api/v1/providers")
async def list_providers() -> dict:
    """Providers, defaults, `.env` availability booleans and masked fingerprints.

    `get_settings()` re-reads `.env` when the file changes, so keys edited while
    uvicorn is running are reflected here (and in the digest pipeline) at once.
    """
    settings = get_settings()
    return {
        "providers": [p.value for p in LLMProviderEnum],
        "default_provider": settings.default_llm_provider,
        "default_model": settings.default_llm_model,
        # Booleans only — never expose the secret values themselves.
        "env_status": settings.env_status(),
        # Masked fingerprints (e.g. "sk-or-v…4a2b") so the dashboard can show
        # whether a UI override or the server `.env` key is actually active.
        "env_fingerprints": settings.provider_key_fingerprints(),
    }


@app.get("/api/v1/stats")
async def get_stats(request: Request) -> dict:
    """Aggregate telemetry: requests, tokens, USD cost, dispatch counts."""
    stats = await request.app.state.db.aggregate_stats()
    stats["by_provider"] = await request.app.state.db.stats_by_provider()
    return stats


@app.get("/api/v1/system/check-updates", response_model=list[DependencyUpdateStatus])
async def system_check_updates(request: Request) -> list[DependencyUpdateStatus]:
    """Compare installed packages with latest stable PyPI releases."""
    return await check_dependency_updates(http_client=request.app.state.http_client)
