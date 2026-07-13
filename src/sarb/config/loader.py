"""Settings construction and the ambient-settings helper.

Two ways to get a :class:`Settings`:

:func:`load_settings`
    Builds a fresh, fully-validated instance. Pure — same inputs, same output,
    no global state. This is what application entrypoints call, and what the DI
    container registers as a singleton.

:func:`current_settings`
    Reads the settings the process has already installed. A convenience for code
    too far from the composition root to have them injected. Backed by a
    :class:`ContextVar`, so :func:`use_settings` can swap them for the duration
    of a test without leaking into neighbouring tests or tasks.

Injected settings beat ambient settings. :func:`current_settings` exists because
the alternative — a module-level ``settings = Settings()`` evaluated at import
time — makes every importing module unusable without a valid environment, which
is the single most common way a config layer becomes untestable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sarb.config.models import Settings
from sarb.config.sources import FileContext, file_context
from sarb.errors import ConfigurationError

__all__ = ["current_settings", "load_settings", "use_settings"]

#: Environment variables read *before* settings exist, to find the settings.
_CONFIG_DIR_ENV = "SARB_CONFIG_DIR"
_PROFILE_ENV = "SARB_PROFILE"

_active: ContextVar[Settings | None] = ContextVar("sarb_active_settings", default=None)


def load_settings(
    *,
    config_dir: str | Path | None = None,
    profile: str | None = None,
    env_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Build a validated :class:`Settings` from every configured layer.

    Args:
        config_dir: Directory holding ``base.yaml`` and profile files. Falls back
            to ``$SARB_CONFIG_DIR``; when neither is set the YAML layer is
            skipped entirely and config comes from env and defaults alone.
        profile: Profile to overlay, e.g. ``"production"``. Falls back to
            ``$SARB_PROFILE``.
        env_file: A ``.env`` file to read. Ignored if it does not exist.
        overrides: Highest-priority values, applied over everything else. Used by
            tests and by callers embedding the framework in a larger app.
        environ: Environment used for ``${VAR}`` expansion in YAML. Defaults to
            the real process environment; injectable for tests.

    Returns:
        A frozen, fully-validated settings object.

    Raises:
        ConfigurationError: If any layer is malformed or any field fails
            validation. The message lists every offending field path at once,
            rather than making the operator fix them one boot at a time.

    Example:
        >>> settings = load_settings(config_dir="config", profile="production")
        >>> settings.component("llms").provider
        'anthropic'
    """
    ctx = FileContext(
        config_dir=_resolve_config_dir(config_dir),
        profile=profile or os.environ.get(_PROFILE_ENV) or None,
        environ=dict(os.environ) if environ is None else environ,
    )

    token = file_context.set(ctx)
    try:
        return Settings(
            _env_file=_resolve_env_file(env_file),
            **(overrides or {}),
        )
    except ValidationError as exc:
        raise ConfigurationError(
            _format_validation_error(exc, ctx),
            context={
                "config_dir": str(ctx.config_dir) if ctx.config_dir else None,
                "profile": ctx.profile,
                "error_count": exc.error_count(),
            },
            cause=exc,
        ) from exc
    finally:
        file_context.reset(token)


def _resolve_config_dir(explicit: str | Path | None) -> Path | None:
    """Pick the config directory from the argument, then the environment."""
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(_CONFIG_DIR_ENV)
    return Path(from_env) if from_env else None


def _resolve_env_file(explicit: str | Path | None) -> Path | None:
    """Return the ``.env`` path only when it exists, so a default is harmless."""
    if explicit is None:
        return None
    path = Path(explicit)
    return path if path.is_file() else None


def _format_validation_error(exc: ValidationError, ctx: FileContext) -> str:
    """Render a Pydantic error as operator-facing guidance.

    A stock Pydantic traceback tells a developer what a Python field is called.
    An operator needs to know which YAML key, or which environment variable, to
    change — so every field path is also rendered in both of those forms.
    """
    lines = [f"Invalid configuration ({exc.error_count()} problem(s)):"]
    for err in exc.errors():
        path = ".".join(str(p) for p in err["loc"])
        env_var = "SARB_" + "__".join(str(p) for p in err["loc"]).upper()
        lines.append(f"  - {path}: {err['msg']}  [env: {env_var}]")

    where = str(ctx.config_dir) if ctx.config_dir else "<no config dir>"
    lines.append(f"  config_dir={where} profile={ctx.profile or '<none>'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ambient settings
# --------------------------------------------------------------------------- #


def current_settings() -> Settings:
    """Return the settings installed for the current context.

    Returns:
        The active settings.

    Raises:
        ConfigurationError: If no settings have been installed. Prefer injecting
            :class:`Settings` through the DI container; reach for this only where
            injection is genuinely impractical.
    """
    settings = _active.get()
    if settings is None:
        raise ConfigurationError(
            "No settings are installed in this context. Call load_settings() at your "
            "entrypoint and install them with use_settings(), or inject Settings via "
            "the DI container."
        )
    return settings


@contextmanager
def use_settings(settings: Settings) -> Iterator[Settings]:
    """Install ``settings`` as the ambient settings for the enclosing block.

    Args:
        settings: The settings to install.

    Yields:
        The same settings, for convenience.

    Example:
        >>> with use_settings(load_settings(overrides={"app": {"debug": True}})) as s:
        ...     assert s.app.debug
    """
    token = _active.set(settings)
    try:
        yield settings
    finally:
        _active.reset(token)
