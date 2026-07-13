"""Tests for the component registry."""

from __future__ import annotations

import pytest

from sarb.contracts import Scope
from sarb.errors import ProviderNotFoundError, RegistryError
from sarb.plugins import ComponentContext, ComponentRegistry


def _factory(ctx: ComponentContext) -> str:
    return f"built {ctx.path}"


class TestRegistration:
    def test_add_then_lookup(self):
        registry = ComponentRegistry()
        registry.add("llms", "anthropic", _factory, summary="Claude models")

        registration = registry.lookup("llms", "anthropic")

        assert registration.key == "anthropic"
        assert registration.family == "llms"
        assert registration.scope is Scope.SINGLETON  # right default for pooled clients
        assert registration.summary == "Claude models"
        assert registration.path == "llms.anthropic"

    def test_duplicate_registration_is_refused(self):
        registry = ComponentRegistry()
        registry.add("llms", "anthropic", _factory, source="builtin")

        with pytest.raises(RegistryError) as exc_info:
            registry.add("llms", "anthropic", _factory, source="acme-plugin")

        message = str(exc_info.value)
        assert "builtin" in message  # names the incumbent
        assert "replace=True" in message  # and the way out

    def test_override_is_allowed_when_deliberate(self):
        registry = ComponentRegistry()
        registry.add("llms", "anthropic", _factory, source="builtin")

        def better(ctx: ComponentContext) -> str:
            return "patched"

        registry.add("llms", "anthropic", better, source="acme-plugin", replace=True)

        assert registry.lookup("llms", "anthropic").source == "acme-plugin"

    def test_same_key_in_different_families_does_not_collide(self):
        registry = ComponentRegistry()
        registry.add("llms", "openai", _factory)
        registry.add("embedders", "openai", _factory)  # legitimately both exist

        assert len(registry) == 2

    def test_scope_is_carried_through(self):
        registry = ComponentRegistry()
        registry.add("memories", "buffer", _factory, scope=Scope.SCOPED)

        assert registry.lookup("memories", "buffer").scope is Scope.SCOPED


class TestLookupFailure:
    def test_unknown_provider_lists_what_is_registered(self):
        registry = ComponentRegistry()
        registry.add("llms", "anthropic", _factory)
        registry.add("llms", "openai", _factory)

        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.lookup("llms", "anthropc")  # typo, as it happens in real life

        message = str(exc_info.value)
        assert "anthropc" in message
        # The error answers the question it raises.
        assert "anthropic" in message
        assert "openai" in message

    def test_unknown_provider_in_empty_family_says_so(self):
        registry = ComponentRegistry()

        with pytest.raises(ProviderNotFoundError, match="none registered"):
            registry.lookup("vector_stores", "qdrant")


class TestIntrospection:
    @pytest.fixture
    def registry(self) -> ComponentRegistry:
        registry = ComponentRegistry()
        registry.add("llms", "openai", _factory)
        registry.add("llms", "anthropic", _factory)
        registry.add("vector_stores", "qdrant", _factory)
        return registry

    def test_providers_are_sorted(self, registry: ComponentRegistry):
        assert registry.providers("llms") == ("anthropic", "openai")

    def test_providers_of_unknown_family_is_empty(self, registry: ComponentRegistry):
        assert registry.providers("nope") == ()

    def test_families(self, registry: ComponentRegistry):
        assert registry.families() == ("llms", "vector_stores")

    def test_supports_accepts_a_family_name(self, registry: ComponentRegistry):
        assert registry.supports("llms", "openai")
        assert not registry.supports("llms", "gemini")

    def test_registrations_are_ordered(self, registry: ComponentRegistry):
        paths = [r.path for r in registry.registrations()]

        assert paths == ["llms.anthropic", "llms.openai", "vector_stores.qdrant"]
