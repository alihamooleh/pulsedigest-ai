"""PulseDigest AI — async SQLite (aiosqlite) persistence for digests & telemetry."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS digest_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    email_sent INTEGER NOT NULL DEFAULT 0,
    discord_sent INTEGER NOT NULL DEFAULT 0,
    summary_report TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_digest_logs_created_at
    ON digest_logs (created_at);
"""


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for log rows."""
    return datetime.now(UTC).isoformat()


class Database:
    """Thin async wrapper around `aiosqlite` for digest logs & aggregate stats."""

    def __init__(self, db_path: str | Path = "pulsedigest.db") -> None:
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the connection and ensure the schema exists."""
        if self._conn is not None:
            return
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.executescript(_SCHEMA_V1)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the connection safely (idempotent)."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() must be awaited first")
        return self._conn

    async def log_digest(
        self,
        *,
        topic: str,
        provider: str,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        email_sent: bool = False,
        discord_sent: bool = False,
        summary_report: str,
    ) -> int:
        """Insert one digest execution row and return its id."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            INSERT INTO digest_logs (
                created_at, topic, provider, model_name,
                prompt_tokens, completion_tokens, total_tokens,
                estimated_cost_usd, email_sent, discord_sent, summary_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                topic,
                provider,
                model_name,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                estimated_cost_usd,
                int(email_sent),
                int(discord_sent),
                summary_report,
            ),
        )
        await conn.commit()
        row_id = int(cursor.lastrowid or 0)
        logger.info("Logged digest id=%s topic=%r", row_id, topic)
        return row_id

    async def get_digest(self, digest_id: int) -> dict[str, Any] | None:
        """Fetch one digest log as a dict, or `None` when missing."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM digest_logs WHERE id = ?", (digest_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def aggregate_stats(self) -> dict[str, float]:
        """Return telemetry aggregates for the dashboard bar."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT
                COUNT(*)                              AS total_requests,
                COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                COALESCE(SUM(prompt_tokens), 0)       AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0)   AS total_completion_tokens,
                COALESCE(SUM(estimated_cost_usd), 0)  AS total_cost_usd,
                COALESCE(SUM(email_sent), 0)          AS emails_sent,
                COALESCE(SUM(discord_sent), 0)        AS discord_sent,
                COALESCE(AVG(estimated_cost_usd), 0)  AS avg_cost_usd
            FROM digest_logs
            """
        )
        row = await cursor.fetchone()
        data = dict(row) if row is not None else {}
        return {
            "total_requests": int(data.get("total_requests", 0)),
            "total_tokens": int(data.get("total_tokens", 0)),
            "total_prompt_tokens": int(data.get("total_prompt_tokens", 0)),
            "total_completion_tokens": int(data.get("total_completion_tokens", 0)),
            "total_cost_usd": float(data.get("total_cost_usd", 0.0)),
            "avg_cost_usd": float(data.get("avg_cost_usd", 0.0)),
            "emails_sent": int(data.get("emails_sent", 0)),
            "discord_sent": int(data.get("discord_sent", 0)),
        }

    async def stats_by_provider(self) -> list[dict[str, Any]]:
        """Per-provider breakdown of requests, tokens, and cost."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT
                provider,
                COUNT(*)                              AS total_requests,
                COALESCE(SUM(total_tokens), 0)        AS total_tokens,
                COALESCE(SUM(estimated_cost_usd), 0)  AS total_cost_usd
            FROM digest_logs
            GROUP BY provider
            ORDER BY total_requests DESC
            """
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
