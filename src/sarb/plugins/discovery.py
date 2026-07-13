"""Finding plugins, and deciding which of them are allowed to run.

A plugin advertises itself on the ``sarb.plugins`` entry-point group::

    # in the plugin's pyproject.toml
    [project.entry-points."sarb.plugins"]
    acme = "acme.sarb:register"

    # in acme/sarb.py
    from sarb.plugins import ComponentRegistry

    def register(registry: ComponentRegistry) -> None:
        registry.add("llms", "acme", build_acme, summary="ACME's hosted model")

Loading a plugin means importing third-party code and calling it. That is a
capability, so it is governed:

* ``plugins.enabled: false`` turns discovery off entirely.
* ``plugins.allowlist`` restricts loading to named plugins. A deployment that
  must not execute arbitrary installed code sets this, and then an attacker who
  manages to get a package installed still cannot get it *loaded*.
* ``plugins.denylist`` skips named plugins.

A plugin that fails to load is a hard error, not a warning. The alternative — a
process that boots "successfully" with half its providers missing, then fails on
the first request that needs one — is strictly worse than not booting.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib.metadata import EntryPoint, entry_points

from sarb.config import PluginSettings, Settings
from sarb.errors import PluginError
from sarb.plugins import builtins
from sarb.plugins.registry import ComponentRegistry

__all__ = [
    "ENTRY_POINT_GROUP",
    "build_registry",
    "load_plugins",
]

#: The entry-point group third-party packages advertise themselves on.
ENTRY_POINT_GROUP = "sarb.plugins"

#: Injectable for tests, which must not have to install a package to exercise
#: discovery. Production passes nothing and gets the real installed set.
type EntryPointProvider = Callable[[], Iterable[EntryPoint]]


def _installed_entry_points() -> Iterable[EntryPoint]:
    """Every entry point advertised on the group by installed distributions."""
    return entry_points(group=ENTRY_POINT_GROUP)


def build_registry(
    settings: Settings,
    *,
    entry_point_provider: EntryPointProvider | None = None,
) -> ComponentRegistry:
    """Assemble the registry a container will resolve against.

    Built-in adapters are registered first, then plugins — so a plugin can
    deliberately override a built-in with ``replace=True``, and cannot do so by
    accident.

    Args:
        settings: Application settings; ``settings.plugins`` governs discovery.
        entry_point_provider: Override the source of entry points. For tests.

    Returns:
        A populated registry.

    Raises:
        PluginError: If any permitted plugin fails to import or register.
    """
    registry = ComponentRegistry()
    builtins.register(registry)
    load_plugins(registry, settings.plugins, entry_point_provider=entry_point_provider)
    return registry


def load_plugins(
    registry: ComponentRegistry,
    config: PluginSettings,
    *,
    entry_point_provider: EntryPointProvider | None = None,
) -> tuple[str, ...]:
    """Discover, filter, and run every permitted plugin's registrar.

    Args:
        registry: The registry each plugin will populate.
        config: Discovery policy — enabled, allowlist, denylist.
        entry_point_provider: Override the source of entry points. For tests.

    Returns:
        The names of the plugins that were loaded, in load order.

    Raises:
        PluginError: If a permitted plugin cannot be imported, does not resolve
            to a callable, or raises while registering. The plugin's name is in
            the message: with a dozen plugins installed, "a plugin failed" is not
            an actionable sentence.
    """
    if not config.enabled:
        return ()

    provider = entry_point_provider or _installed_entry_points
    loaded: list[str] = []

    for entry_point in sorted(provider(), key=lambda ep: ep.name):
        if not _is_permitted(entry_point.name, config):
            continue
        _run_registrar(registry, entry_point)
        loaded.append(entry_point.name)

    return tuple(loaded)


def _is_permitted(name: str, config: PluginSettings) -> bool:
    """Whether policy allows this plugin to load.

    The allowlist, when non-empty, is exclusive: anything not named is refused.
    The denylist is then applied on top, so a name in both is refused — the safe
    reading of a contradictory policy.
    """
    if config.allowlist and name not in config.allowlist:
        return False
    return name not in config.denylist


def _run_registrar(registry: ComponentRegistry, entry_point: EntryPoint) -> None:
    """Import one plugin and let it register its components.

    Every failure mode is reported with the plugin's name attached, because the
    operator's next question is always "which one?".
    """
    try:
        registrar = entry_point.load()
    except Exception as exc:
        raise PluginError(
            f"Plugin {entry_point.name!r} could not be imported from {entry_point.value!r}: {exc}",
            context={"plugin": entry_point.name, "target": entry_point.value},
            cause=exc,
        ) from exc

    if not callable(registrar):
        raise PluginError(
            f"Plugin {entry_point.name!r} must point at a callable taking a "
            f"ComponentRegistry, but {entry_point.value!r} is "
            f"{type(registrar).__name__}.",
            context={"plugin": entry_point.name, "target": entry_point.value},
        )

    try:
        registrar(registry)
    except Exception as exc:
        raise PluginError(
            f"Plugin {entry_point.name!r} failed while registering its components: {exc}",
            context={"plugin": entry_point.name, "target": entry_point.value},
            cause=exc,
        ) from exc
