"""Unit tests for the dependency watchdog (`updater.py`) — PyPI mocked."""

from __future__ import annotations

import importlib.metadata

import httpx
import pytest

from schemas import DependencyUpdateStatus
from updater import (
    PYPI_JSON_URL,
    _latest_stable_from_pypi,
    check_dependency_updates,
    is_stable_version,
    notify_updates_if_available,
)


def pypi_json(releases: dict) -> httpx.Response:
    return httpx.Response(200, json={"releases": releases, "info": {"version": "1.0.0"}})


class TestIsStableVersion:
    @pytest.mark.parametrize("version", ["1.2.3", "0.143.2", "3.0.0", "10", "1.0.0.post1"])
    def test_stable_versions_accepted(self, version: str) -> None:
        assert is_stable_version(version) is True

    @pytest.mark.parametrize("version", ["1.0.0rc1", "1.0.0b1", "1.0.0a2", "1.0.dev0", "garbage"])
    def test_pre_releases_and_garbage_rejected(self, version: str) -> None:
        assert is_stable_version(version) is False

    def test_latest_stable_skips_pre_releases(self) -> None:
        latest = _latest_stable_from_pypi(
            {"releases": {"1.0.0": [], "2.0.0": [], "3.0.0rc1": []}}
        )
        assert latest == "2.0.0"


@pytest.fixture()
def fake_installed(monkeypatch: pytest.MonkeyPatch):
    """Fake importlib.metadata.version for watched packages."""
    versions = {
        "fastapi": "0.99.0",
        "langchain-core": "0.3.0",
        "langchain-openai": "0.2.0",
        "pydantic": "2.14.1",
        "aiosqlite": "0.22.1",
    }
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: versions.get(name, "")
    )
    return versions


class TestCheckDependencyUpdates:
    async def test_update_flag_and_payload(self, monkeypatch, fake_installed) -> None:
        payload = {"releases": {"0.99.0": [], "1.0.0": [], "1.1.0b1": []}}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL(PYPI_JSON_URL.format(package="fastapi"))
            return httpx.Response(200, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            results = await check_dependency_updates(
                http_client=client, packages=("fastapi",)
            )
        finally:
            await client.aclose()

        assert len(results) == 1
        status = results[0]
        assert isinstance(status, DependencyUpdateStatus)
        assert status.package_name == "fastapi"
        assert status.installed_version == "0.99.0"
        assert status.latest_stable_version == "1.0.0"
        assert status.update_available is True
        assert "pypi.org/project/fastapi/" in status.release_url

    async def test_fully_up_to_date(self, fake_installed) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"releases": {"2.14.1": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            results = await check_dependency_updates(
                http_client=client, packages=("pydantic",)
            )
        finally:
            await client.aclose()
        assert results[0].update_available is False

    async def test_uninstalled_package_not_flagged(self, fake_installed) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"releases": {"9.9.9": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            results = await check_dependency_updates(
                http_client=client, packages=("totally-missing-package",)
            )
        finally:
            await client.aclose()
        assert results[0].installed_version == ""
        assert results[0].update_available is False

    async def test_http_error_skips_package(self, fake_installed) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="ping down")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            results = await check_dependency_updates(
                http_client=client, packages=("fastapi",)
            )
        finally:
            await client.aclose()
        assert results == []


class TestNotifyUpdates:
    async def test_discord_fired_when_updates_exist(self, monkeypatch, fake_installed) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "fastapi" in request.url.path:
                return httpx.Response(200, json={"releases": {"1.0.0": []}})
            return httpx.Response(200, json={"releases": {"0.1.0": []}})

        sent: list[dict] = []

        async def fake_discord_alert(*args, **kwargs) -> bool:
            sent.append(kwargs)
            return True

        monkeypatch.setattr("updater.send_discord_alert", fake_discord_alert)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            count = await notify_updates_if_available(
                "https://discord.example/webhook", http_client=client
            )
        finally:
            await client.aclose()
        assert count == 1
        assert sent[0]["webhook_url"] == "https://discord.example/webhook"

    async def test_silent_when_up_to_date(
        self, monkeypatch: pytest.MonkeyPatch, fake_installed
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"releases": {"0.99.0": [], "1.0.0": []}})

        # All watched packages pinned to their latest → no notification attempted
        fake_installed["fastapi"] = "1.0.0"
        fake_installed["langchain-core"] = "1.0.0"
        fake_installed["langchain-openai"] = "1.0.0"
        fake_installed["pydantic"] = "1.0.0"
        fake_installed["aiosqlite"] = "1.0.0"

        async def fail_alert(*args, **kwargs) -> bool:
            pytest.fail("discord must not be notified when up to date")

        monkeypatch.setattr("updater.send_discord_alert", fail_alert)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            count = await notify_updates_if_available(None, http_client=client)
        finally:
            await client.aclose()
        assert count == 0
