"""PulseDigest AI — self-updating dependency watchdog.

Checks installed versions of core packages against PyPI's latest *stable*
release (pre-releases `rc`/`b`/`a`/`dev` excluded) and optionally alerts
Discord when stable updates are available.
"""

from __future__ import annotations

import importlib.metadata
import logging

import httpx
from packaging.version import InvalidVersion, Version

from schemas import DependencyUpdateStatus
from tools import send_discord_alert

logger = logging.getLogger(__name__)

WATCHED_PACKAGES: tuple[str, ...] = (
    "fastapi",
    "langchain-core",
    "langchain-openai",
    "pydantic",
    "aiosqlite",
)
PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"


def is_stable_version(version_str: str) -> bool:
    """True for stable releases only — rejects rc/b/a/dev pre-releases."""
    try:
        parsed = Version(version_str)
    except InvalidVersion:
        return False
    return parsed.pre is None and parsed.dev is None


def _latest_stable_from_pypi(payload: dict) -> str | None:
    """Highest stable version present in a PyPI JSON releases payload."""
    stable: list[Version] = []
    for candidate in list(payload.get("releases", {})):
        if is_stable_version(candidate):
            stable.append(Version(candidate))
    return str(max(stable)) if stable else None


def _installed_version(package_name: str) -> str:
    """Installed version via importlib.metadata ('' when not installed)."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


async def check_dependency_updates(
    http_client: httpx.AsyncClient | None = None,
    packages: tuple[str, ...] | None = None,
) -> list[DependencyUpdateStatus]:
    """Compare installed vs latest stable PyPI versions for watched packages."""
    packages = packages or WATCHED_PACKAGES
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=30.0)
    results: list[DependencyUpdateStatus] = []
    try:
        for package_name in packages:
            installed = _installed_version(package_name)
            release_url = f"https://pypi.org/project/{package_name}/"
            try:
                response = await client.get(PYPI_JSON_URL.format(package=package_name))
            except httpx.HTTPError as exc:
                logger.warning("PyPI unreachable for %s: %s", package_name, exc)
                continue
            if response.status_code >= 400:
                logger.warning(
                    "PyPI lookup failed for %s (HTTP %s)", package_name, response.status_code
                )
                continue
            try:
                payload = response.json()
            except ValueError:
                logger.warning("PyPI returned non-JSON payload for %s.", package_name)
                continue
            latest_stable = _latest_stable_from_pypi(payload) or payload.get("info", {}).get("version")
            if not latest_stable:
                continue
            update_available = bool(installed) and Version(latest_stable) > Version(installed)
            results.append(
                DependencyUpdateStatus(
                    package_name=package_name,
                    installed_version=installed,
                    latest_stable_version=latest_stable,
                    update_available=update_available,
                    release_url=release_url,
                )
            )
    finally:
        if owned:
            await client.aclose()
    return results


async def notify_updates_if_available(
    webhook_url: str | None,
    http_client: httpx.AsyncClient,
) -> int:
    """When stable updates exist, post one consolidated Discord embed.

    Returns the number of packages with a stable update available.
    """
    updates = await check_dependency_updates(http_client=http_client)
    available = [u for u in updates if u.update_available]
    if not available:
        logger.info("All watched dependencies are up to date.")
        return 0

    summary = "\n".join(
        f"**{u.package_name}**: `{u.installed_version}` → "
        f"`{u.latest_stable_version}`"
        for u in available
    )
    sent = await send_discord_alert(
        webhook_url=webhook_url,
        title=f"🚀 PulseDigest AI: {len(available)} dependency update(s) available",
        summary=summary,
        cost_usd=0.0,
        http_client=http_client,
    )
    if not sent:
        logger.warning("Dependency update notification could not be delivered.")
    return len(available)
