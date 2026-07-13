"""Tests for plugin discovery and the policy that governs it.

Entry points are injected rather than installed: exercising the allowlist should
not require building and installing a package.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import EntryPoint

import pytest

from sarb.config import PluginSettings, load_settings
from sarb.errors import PluginError
from sarb.plugins import ComponentContext, ComponentRegistry, build_registry, load_plugins
from sarb.plugins.discovery import EntryPointProvider

# --------------------------------------------------------------------------- #
# Test doubles, referenced by the fake entry points below.
# --------------------------------------------------------------------------- #


def _factory(ctx: ComponentContext) -> str:
    return ctx.path


def register_acme(registry: ComponentRegistry) -> None:
    registry.add("llms", "acme", _factory, source="acme")


def register_globex(registry: ComponentRegistry) -> None:
    registry.add("llms", "globex", _factory, source="globex")


def register_broken(registry: ComponentRegistry) -> None:
    raise RuntimeError("plugin author made a mistake")


NOT_CALLABLE = "this is a string, not a registrar"


def _entry_point(name: str, attr: str) -> EntryPoint:
    """An entry point aimed at something in this test module."""
    return EntryPoint(name=name, value=f"{__name__}:{attr}", group="sarb.plugins")


def _provider(*eps: EntryPoint) -> EntryPointProvider:
    """An entry-point source returning exactly these, instead of what's installed."""

    def provide() -> Iterable[EntryPoint]:
        return eps

    return provide


@pytest.fixture
def registry() -> ComponentRegistry:
    return ComponentRegistry()


class TestLoading:
    def test_loads_a_plugin_and_registers_its_components(self, registry: ComponentRegistry):
        loaded = load_plugins(
            registry,
            PluginSettings(),
            entry_point_provider=_provider(_entry_point("acme", "register_acme")),
        )

        assert loaded == ("acme",)
        assert registry.supports("llms", "acme")
        assert registry.lookup("llms", "acme").source == "acme"

    def test_loads_plugins_in_a_deterministic_order(self, registry: ComponentRegistry):
        # Registration order decides who wins a `replace=True`, so it must not
        # depend on however the metadata happened to be enumerated.
        loaded = load_plugins(
            registry,
            PluginSettings(),
            entry_point_provider=_provider(
                _entry_point("globex", "register_globex"),
                _entry_point("acme", "register_acme"),
            ),
        )

        assert loaded == ("acme", "globex")


class TestPolicy:
    def test_disabled_loads_nothing(self, registry: ComponentRegistry):
        loaded = load_plugins(
            registry,
            PluginSettings(enabled=False),
            entry_point_provider=_provider(_entry_point("acme", "register_acme")),
        )

        assert loaded == ()
        assert len(registry) == 0

    def test_allowlist_is_exclusive(self, registry: ComponentRegistry):
        loaded = load_plugins(
            registry,
            PluginSettings(allowlist=("acme",)),
            entry_point_provider=_provider(
                _entry_point("acme", "register_acme"),
                _entry_point("globex", "register_globex"),
            ),
        )

        assert loaded == ("acme",)
        assert not registry.supports("llms", "globex")

    def test_denylist_skips(self, registry: ComponentRegistry):
        loaded = load_plugins(
            registry,
            PluginSettings(denylist=("globex",)),
            entry_point_provider=_provider(
                _entry_point("acme", "register_acme"),
                _entry_point("globex", "register_globex"),
            ),
        )

        assert loaded == ("acme",)

    def test_denylist_wins_over_allowlist(self, registry: ComponentRegistry):
        # A contradictory policy resolves to the safe reading: don't load it.
        loaded = load_plugins(
            registry,
            PluginSettings(allowlist=("acme",), denylist=("acme",)),
            entry_point_provider=_provider(_entry_point("acme", "register_acme")),
        )

        assert loaded == ()


class TestFailure:
    def test_a_registrar_that_raises_names_the_plugin(self, registry: ComponentRegistry):
        with pytest.raises(PluginError) as exc_info:
            load_plugins(
                registry,
                PluginSettings(),
                entry_point_provider=_provider(_entry_point("broken", "register_broken")),
            )

        assert "broken" in str(exc_info.value)
        assert exc_info.value.context["plugin"] == "broken"
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_an_unimportable_plugin_names_the_plugin(self, registry: ComponentRegistry):
        bad = EntryPoint(name="ghost", value="no.such.module:register", group="sarb.plugins")

        with pytest.raises(PluginError, match="ghost"):
            load_plugins(registry, PluginSettings(), entry_point_provider=_provider(bad))

    def test_a_non_callable_target_is_refused(self, registry: ComponentRegistry):
        with pytest.raises(PluginError, match="callable"):
            load_plugins(
                registry,
                PluginSettings(),
                entry_point_provider=_provider(_entry_point("weird", "NOT_CALLABLE")),
            )

    def test_a_failing_plugin_stops_the_boot(self, registry: ComponentRegistry):
        # Booting "successfully" with half the providers missing, then failing on
        # the first request that needs one, is strictly worse than not booting.
        with pytest.raises(PluginError):
            load_plugins(
                registry,
                PluginSettings(),
                entry_point_provider=_provider(
                    _entry_point("acme", "register_acme"),
                    _entry_point("broken", "register_broken"),
                ),
            )


class TestBuildRegistry:
    def test_builds_from_settings(self):
        settings = load_settings()

        registry = build_registry(
            settings,
            entry_point_provider=_provider(_entry_point("acme", "register_acme")),
        )

        assert registry.supports("llms", "acme")

    def test_honours_plugin_settings(self):
        settings = load_settings(overrides={"plugins": {"enabled": False}})

        registry = build_registry(
            settings,
            entry_point_provider=_provider(_entry_point("acme", "register_acme")),
        )

        # Discovery is off, so the plugin is not there. The built-ins still are:
        # turning off *third-party* plugin discovery must not disable the framework's
        # own adapters, which is what an operator hardening a deployment expects.
        assert not registry.supports("llms", "acme")
        assert registry.supports("llms", "anthropic")

    def test_the_builtins_are_always_registered(self):
        registry = build_registry(load_settings(), entry_point_provider=_provider())

        assert set(registry.providers("llms")) == {
            "anthropic",
            "openai",
            "gemini",
            "ollama",
            "fake",
        }

    def test_the_real_entry_point_scan_does_not_explode(self):
        # No plugins are installed, so this must produce a built-ins-only registry
        # rather than raising. Guards the default production path.
        assert isinstance(build_registry(load_settings()), ComponentRegistry)
