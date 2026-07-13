"""The YAML configuration layer.

Adds file-based config beneath environment variables, with two properties that
matter in production:

**Profiles.** ``base.yaml`` holds what is true everywhere; ``<profile>.yaml``
holds what differs. They are deep-merged, so a staging file that sets one field
of one component does not have to restate the other twenty.

**Interpolation.** ``${VAR}`` and ``${VAR:-default}` are expanded from the
process environment while parsing, so a YAML file can reference a secret without
containing it. A file that is missing a required variable fails at load, not on
first use.

The profile and config directory are passed through a :class:`ContextVar` rather
than mutating ``os.environ``. pydantic-settings constructs sources itself and
gives no way to hand them arguments; a context variable threads the values
through without leaving global state behind or making concurrent loads race.
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from sarb.errors import ConfigurationError

__all__ = ["FileContext", "YamlSettingsSource", "deep_merge", "expand_env", "file_context"]

#: ``${NAME}`` or ``${NAME:-default}``. A doubled ``$$`` escapes the sequence,
#: which keeps literal ``${...}`` (e.g. inside a prompt template) expressible.
_PLACEHOLDER = re.compile(
    r"""
    (?P<escape>\$\$)                      # $$  -> literal $
    |
    \$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)  # variable name
        (?: :- (?P<default>[^}]*) )?      # optional :-default
    \}
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class FileContext:
    """Where the YAML layer should read from.

    Attributes:
        config_dir: Directory holding ``base.yaml`` and profile files. ``None``
            disables the YAML layer entirely, which is the right default for
            library consumers and for tests.
        profile: Profile name, e.g. ``"production"``. Selects ``<profile>.yaml``.
        environ: Mapping used for ``${VAR}`` expansion. Injectable so tests do
            not have to mutate the real process environment.
    """

    config_dir: Path | None = None
    profile: str | None = None
    environ: dict[str, str] = field(default_factory=lambda: dict(os.environ))


#: Active file context. Set by :func:`sarb.config.load_settings` for the duration
#: of one load. The default is ``None`` rather than a shared ``FileContext()``:
#: a single mutable default object would be captured by every load in the
#: process, so one caller's environment could bleed into another's.
file_context: ContextVar[FileContext | None] = ContextVar("sarb_file_context", default=None)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Loads ``base.yaml`` + ``<profile>.yaml`` from the active file context."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._context = file_context.get() or FileContext()

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source produces the whole mapping in :meth:`__call__`."""
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        """Read, merge, and interpolate the configured YAML files."""
        ctx = self._context
        config_dir = ctx.config_dir
        if config_dir is None:
            return {}

        if not config_dir.is_dir():
            raise ConfigurationError(
                f"Config directory does not exist: {config_dir}",
                context={"config_dir": str(config_dir)},
            )

        merged: dict[str, Any] = {}
        for path in self._candidate_files(config_dir, ctx.profile):
            merged = deep_merge(merged, _read_yaml(path))

        # expand_env is structure-preserving, so a mapping in is a mapping out.
        return cast("dict[str, Any]", expand_env(merged, ctx.environ))

    @staticmethod
    def _candidate_files(config_dir: Path, profile: str | None) -> list[Path]:
        """Return the files to merge, lowest precedence first.

        A missing ``base.yaml`` is tolerated (a deployment may configure purely
        by profile). A missing *profile* file is not: asking for ``production``
        and silently getting development defaults is exactly the failure this
        framework should refuse to have.
        """
        files: list[Path] = []

        base = config_dir / "base.yaml"
        if base.is_file():
            files.append(base)

        if profile:
            profile_file = config_dir / f"{profile}.yaml"
            if not profile_file.is_file():
                available = sorted(p.stem for p in config_dir.glob("*.yaml"))
                raise ConfigurationError(
                    f"Profile {profile!r} has no config file at {profile_file}. "
                    f"Available profiles: {', '.join(available) or '<none>'}",
                    context={"profile": profile, "available": available},
                )
            files.append(profile_file)

        return files

    def __repr__(self) -> str:
        """Identify the source in pydantic-settings error output."""
        return f"{type(self).__name__}(config_dir={self._context.config_dir!r})"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse one YAML file into a mapping.

    Raises:
        ConfigurationError: If the file is unparseable or is not a mapping at the
            top level (a bare list or scalar cannot be merged into settings).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Malformed YAML in {path}: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read config file {path}: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Config file {path} must contain a mapping at the top level, got {type(raw).__name__}",
            context={"path": str(path)},
        )
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested mappings merge key by key; every other type replaces wholesale. Lists
    replace rather than concatenate, because appending is almost never what an
    operator means — a profile that sets ``cors_origins`` intends to *define*
    them, not to add to the base list with no way to remove an entry.

    Args:
        base: The lower-precedence mapping.
        override: The higher-precedence mapping.

    Returns:
        A new merged mapping. Neither input is modified.

    Example:
        >>> deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        {'a': {'x': 1, 'y': 3}}
    """
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = deep_merge(current, value)
        else:
            result[key] = value
    return result


def expand_env(value: Any, environ: dict[str, str]) -> Any:  # noqa: ANN401 - walks arbitrary YAML
    """Recursively expand ``${VAR}`` placeholders in every string in ``value``.

    Args:
        value: Any YAML-derived structure.
        environ: The environment to resolve against.

    Returns:
        The structure with placeholders substituted.

    Raises:
        ConfigurationError: If a referenced variable is unset and no ``:-``
            default was supplied. Failing here means a container missing its API
            key dies at boot with a precise message, instead of serving traffic
            until the first model call.

    Example:
        >>> expand_env({"k": "${HOME}/x"}, {"HOME": "/root"})
        {'k': '/root/x'}
        >>> expand_env("${MISSING:-fallback}", {})
        'fallback'
    """
    if isinstance(value, dict):
        return {k: expand_env(v, environ) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, environ) for v in value]
    if isinstance(value, str):
        return _expand_string(value, environ)
    return value


def _expand_string(raw: str, environ: dict[str, str]) -> str:
    """Substitute every placeholder in one string."""
    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        if match.group("escape"):
            return "$"
        name = match.group("name")
        if name in environ:
            return environ[name]
        default = match.group("default")
        if default is not None:
            return default
        missing.append(name)
        return ""

    expanded = _PLACEHOLDER.sub(substitute, raw)

    if missing:
        raise ConfigurationError(
            "Config references environment variables that are not set: "
            f"{', '.join(sorted(set(missing)))}. Set them, or give them a default "
            "with ${NAME:-value}.",
            context={"missing": sorted(set(missing))},
        )
    return expanded
