"""The composition root: settings plus registry produce a container.

This is the two-line function every entrypoint — the API server, the CLI, a
worker, a test — calls to bring the framework up. It exists so that "how is the
application assembled?" has exactly one answer, and so that the answer is short
enough to read.

    async with bootstrap(profile="production") as container:
        llm = await container.get(LLMS)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sarb.config import Settings, load_settings
from sarb.di.container import Container
from sarb.plugins.discovery import build_registry

__all__ = ["bootstrap", "build_container"]


def build_container(settings: Settings) -> Container:
    """Assemble an application container from settings.

    Discovers plugins, builds the registry, and returns a container ready to
    resolve. Nothing is constructed yet — components are built on first use, so
    an application that never touches its vector store never opens a connection
    to one.

    Args:
        settings: Validated application settings.

    Returns:
        An open root container. The caller owns it and must close it, which is
        what :func:`bootstrap` is for.

    Raises:
        PluginError: If a permitted plugin fails to load.
    """
    return Container(settings, build_registry(settings))


@asynccontextmanager
async def bootstrap(
    *,
    config_dir: str | Path | None = None,
    profile: str | None = None,
    env_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AsyncIterator[Container]:
    """Load settings, assemble the container, and close it on exit.

    Args:
        config_dir: Directory holding ``base.yaml`` and profile files.
        profile: Profile to overlay, e.g. ``"production"``.
        env_file: A ``.env`` file to read.
        overrides: Highest-priority settings values.

    Yields:
        The application container.

    Raises:
        ConfigurationError: If settings are missing or invalid — before anything
            is constructed, and before any port is bound.
        ExceptionGroup: If components fail while shutting down.

    Example:
        >>> async with bootstrap(config_dir="config", profile="production") as c:
        ...     llm = await c.get(LLMS)
    """
    settings = load_settings(
        config_dir=config_dir,
        profile=profile,
        env_file=env_file,
        overrides=overrides,
    )
    async with build_container(settings) as container:
        yield container
