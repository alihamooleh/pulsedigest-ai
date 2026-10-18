"""Unit tests for async SQLite persistence (`database.py`)."""

from __future__ import annotations

import pytest

from database import Database


@pytest.fixture()
async def db(tmp_path) -> Database:  # type: ignore[misc]
    database = Database(db_path=tmp_path / "pulsedigest.db")
    await database.connect()
    yield database
    await database.close()


class TestConnectLifecycle:
    async def test_connect_closes_and_reconnects(self, tmp_path) -> None:
        db = Database(db_path=tmp_path / "a.db")
        await db.connect()
        await db.close()
        await db.connect()
        await db.close()

    async def test_operations_require_connect(self) -> None:
        db = Database()
        with pytest.raises(RuntimeError, match="connect"):
            await db.log_digest(topic="t", provider="openai", model_name="m",
                                summary_report="# r")


class TestLogAndFetch:
    async def test_log_digest_returns_row_id(self, db: Database) -> None:
        row_id = await db.log_digest(
            topic="quantum computing",
            provider="openrouter",
            model_name="openrouter/auto",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.01,
            email_sent=False,
            discord_sent=True,
            summary_report="# Digest\n-yahoo",
        )
        assert row_id >= 1

    async def test_row_isolated(self, tmp_path) -> None:
        db_1 = Database(db_path=tmp_path / "x.db")
        await db_1.connect()
        await db_1.close()  # ensure re-open persistence across connections
        await db_1.connect()
        await db_1.log_digest(
            topic="edge ai", provider="openai", model_name="gpt-5.6",
            summary_report="r",
        )
        row = await db_1.get_digest(1)
        assert row is not None
        assert row["topic"] == "edge ai"
        assert row["provider"] == "openai"
        assert row["model_name"] == "gpt-5.6"
        assert row["total_tokens"] == 0
        assert row["email_sent"] == 0 and row["discord_sent"] == 0
        await db_1.close()

    async def test_get_missing_digest_returns_none(self, db: Database) -> None:
        assert await db.get_digest(999) is None


class TestAggregates:
    async def test_empty_db_stats(self, db: Database) -> None:
        stats = await db.aggregate_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost_usd"] == 0.0

    async def test_aggregate_and_per_provider_stats(self, db: Database) -> None:
        await db.log_digest(
            topic="t1", provider="openrouter", model_name="openrouter/auto",
            prompt_tokens=10, completion_tokens=5, estimated_cost_usd=0.02,
            email_sent=True, summary_report="digest one",
        )
        await db.log_digest(
            topic="t2", provider="openai", model_name="gpt-6",
            prompt_tokens=0, completion_tokens=5, estimated_cost_usd=0.10,
            discord_sent=True, summary_report="digest two",
        )
        stats = await db.aggregate_stats()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 20
        assert stats["total_prompt_tokens"] == 10
        assert stats["total_completion_tokens"] == 10
        assert stats["total_cost_usd"] == pytest.approx(0.12)
        assert stats["avg_cost_usd"] == pytest.approx(0.06)
        assert stats["emails_sent"] == 1
        assert stats["discord_sent"] == 1

        per_provider = await db.stats_by_provider()
        assert {row["provider"] for row in per_provider} == {"openrouter", "openai"}
        openai_row = next(r for r in per_provider if r["provider"] == "openai")
        assert openai_row["total_requests"] == 1
        assert openai_row["total_cost_usd"] == pytest.approx(0.10)
