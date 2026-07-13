"""Tests for the DI container.

The behaviours that matter most here are the ones that fail silently in
production if they regress: singleton identity under concurrency, shutdown order,
and the refusal to hand a request-scoped object to the application root.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import anyio
import pytest

from sarb.config import Settings, load_settings
from sarb.contracts import Family, Scope
from sarb.di import Container
from sarb.errors import ConfigurationError, ProviderNotFoundError, RegistryError
from sarb.plugins import ComponentContext, ComponentRegistry

# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


@runtime_checkable
class Greeter(Protocol):
    """A stand-in port. Structural, like every real port in the framework."""

    def greet(self) -> str: ...


class Hello:
    def __init__(self, name: str = "world") -> None:
        self.name = name
        self.closed = False

    def greet(self) -> str:
        return f"hello {self.name}"

    async def aclose(self) -> None:
        self.closed = True


class NotAGreeter:
    """Registered against the Greeter family, but missing greet()."""

    def hail(self) -> str:
        return "wrong shape"


class Clock(Protocol):
    def now(self) -> int: ...


class FixedClock:
    def now(self) -> int:
        return 42


GREETERS = Family("llms", Greeter)  # reuses the `llms` config family for brevity


def build_hello(ctx: ComponentContext) -> Hello:
    return Hello(ctx.config.options.get("name", "world"))


async def build_hello_async(ctx: ComponentContext) -> Hello:
    await anyio.lowlevel.checkpoint()  # a real factory would open a connection here
    return Hello(ctx.config.options.get("name", "async"))


@pytest.fixture
def registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.add("llms", "hello", build_hello)
    registry.add("llms", "hello-async", build_hello_async)
    registry.add("llms", "wrong-shape", lambda ctx: NotAGreeter())
    registry.add("llms", "scoped-hello", build_hello, scope=Scope.SCOPED)
    registry.add("llms", "transient-hello", build_hello, scope=Scope.TRANSIENT)
    return registry


@pytest.fixture
def settings() -> Settings:
    return load_settings(
        overrides={
            "llms": {
                "default": {"provider": "hello", "options": {"name": "default"}},
                "async": {"provider": "hello-async"},
                "broken": {"provider": "wrong-shape"},
                "scoped": {"provider": "scoped-hello"},
                "transient": {"provider": "transient-hello"},
                "off": {"provider": "hello", "enabled": False},
                "unregistered": {"provider": "nobody-registered-this"},
            }
        }
    )


@pytest.fixture
def container(settings: Settings, registry: ComponentRegistry) -> Container:
    return Container(settings, registry)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #


class TestComponentResolution:
    async def test_resolves_from_config_through_the_registry(self, container: Container):
        greeter = await container.get(GREETERS)

        assert greeter.greet() == "hello default"

    async def test_resolves_by_name(self, container: Container):
        greeter = await container.get(GREETERS, "async")

        assert greeter.greet() == "hello async"  # async factory was awaited

    async def test_singletons_are_cached(self, container: Container):
        first = await container.get(GREETERS)
        second = await container.get(GREETERS)

        assert first is second

    async def test_different_names_are_different_instances(self, container: Container):
        assert await container.get(GREETERS) is not await container.get(GREETERS, "async")

    async def test_unconfigured_component_is_a_config_error(self, container: Container):
        with pytest.raises(ConfigurationError, match="judge"):
            await container.get(GREETERS, "judge")

    async def test_disabled_component_is_refused(self, container: Container):
        with pytest.raises(ConfigurationError, match="disabled"):
            await container.get(GREETERS, "off")

    async def test_unregistered_provider_is_reported(self, container: Container):
        with pytest.raises(ProviderNotFoundError, match="nobody-registered-this"):
            await container.get(GREETERS, "unregistered")

    async def test_an_adapter_of_the_wrong_shape_is_caught_at_wiring_time(
        self, container: Container
    ):
        # Not on the first call in production, three layers deep.
        with pytest.raises(RegistryError, match="does not satisfy Greeter"):
            await container.get(GREETERS, "broken")

    async def test_factory_receives_its_config_and_a_resolver(
        self, settings: Settings, registry: ComponentRegistry
    ):
        seen: list[ComponentContext] = []

        def spy(ctx: ComponentContext) -> Hello:
            seen.append(ctx)
            return Hello()

        registry.add("llms", "spy", spy)
        settings = load_settings(overrides={"llms": {"default": {"provider": "spy"}}})

        async with Container(settings, registry) as container:
            await container.get(GREETERS)

        (ctx,) = seen
        assert ctx.path == "llms.default"
        assert ctx.config.provider == "spy"
        assert ctx.resolver is container  # a factory can resolve its dependencies

    async def test_a_factory_can_resolve_another_component(
        self, settings: Settings, registry: ComponentRegistry
    ):
        async def build_composite(ctx: ComponentContext) -> Greeter:
            inner = await ctx.resolver.resolve(GREETERS, "async")
            return Hello(f"wrapping {inner.greet()}")

        registry.add("llms", "composite", build_composite)
        settings = load_settings(
            overrides={
                "llms": {
                    "default": {"provider": "composite"},
                    "async": {"provider": "hello-async"},
                }
            }
        )

        async with Container(settings, registry) as container:
            assert (await container.get(GREETERS)).greet() == "hello wrapping hello async"

    async def test_a_failing_factory_names_the_component(
        self, settings: Settings, registry: ComponentRegistry
    ):
        def explode(ctx: ComponentContext) -> Hello:
            raise ValueError("no credentials")

        registry.add("llms", "explodes", explode)
        settings = load_settings(overrides={"llms": {"default": {"provider": "explodes"}}})

        with pytest.raises(RegistryError, match=r"llms\.default"):
            await Container(settings, registry).get(GREETERS)


class TestConcurrency:
    async def test_racing_tasks_get_exactly_one_singleton(
        self, settings: Settings, registry: ComponentRegistry
    ):
        # A second HTTP pool created because two requests landed at once is the
        # kind of bug that only appears under load, in production.
        builds = 0

        async def slow_build(ctx: ComponentContext) -> Hello:
            nonlocal builds
            builds += 1
            await anyio.sleep(0.01)  # a real client is opening a connection here
            return Hello()

        registry.add("llms", "slow", slow_build)
        settings = load_settings(overrides={"llms": {"default": {"provider": "slow"}}})
        container = Container(settings, registry)

        results: list[Greeter] = []

        async with anyio.create_task_group() as tg:
            for _ in range(20):
                tg.start_soon(lambda: _collect(container, results))

        assert builds == 1
        assert all(r is results[0] for r in results)


async def _collect(container: Container, out: list[Greeter]) -> None:
    out.append(await container.get(GREETERS))


class TestCycles:
    async def test_a_dependency_cycle_is_reported_not_deadlocked(
        self, settings: Settings, registry: ComponentRegistry
    ):
        async def build_a(ctx: ComponentContext) -> Greeter:
            return await ctx.resolver.resolve(GREETERS, "b")

        async def build_b(ctx: ComponentContext) -> Greeter:
            return await ctx.resolver.resolve(GREETERS, "default")

        registry.add("llms", "a", build_a)
        registry.add("llms", "b", build_b)
        settings = load_settings(
            overrides={
                "llms": {
                    "default": {"provider": "a"},
                    "b": {"provider": "b"},
                }
            }
        )

        with pytest.raises(RegistryError) as exc_info:
            await Container(settings, registry).get(GREETERS)

        message = str(exc_info.value)
        assert "Circular dependency" in message
        assert "llms.default" in message
        assert "llms.b" in message


# --------------------------------------------------------------------------- #
# Scopes and lifecycle
# --------------------------------------------------------------------------- #


class TestScopes:
    async def test_scoped_components_are_per_scope(self, container: Container):
        async with container.scope() as first:
            a = await first.get(GREETERS, "scoped")
            a2 = await first.get(GREETERS, "scoped")

        async with container.scope() as second:
            b = await second.get(GREETERS, "scoped")

        assert a is a2  # cached within the scope
        assert a is not b  # but not across scopes

    async def test_scoped_component_cannot_be_resolved_from_the_root(self, container: Container):
        # The bug this prevents: a per-request object captured by the
        # application container and served to the next user.
        with pytest.raises(RegistryError, match="request-scoped"):
            await container.get(GREETERS, "scoped")

    async def test_scoped_instances_are_closed_when_the_scope_exits(self, container: Container):
        async with container.scope() as scope:
            greeter = await scope.get(GREETERS, "scoped")
            assert not greeter.closed  # type: ignore[attr-defined]

        assert greeter.closed  # type: ignore[attr-defined]

    async def test_singletons_survive_a_scope(self, container: Container):
        async with container.scope() as scope:
            inner = await scope.get(GREETERS)

        outer = await container.get(GREETERS)

        assert inner is outer
        assert not inner.closed  # type: ignore[attr-defined]
        await container.aclose()
        assert inner.closed  # type: ignore[attr-defined]

    async def test_transients_are_never_cached(self, container: Container):
        async with container.scope() as scope:
            first = await scope.get(GREETERS, "transient")
            second = await scope.get(GREETERS, "transient")

        assert first is not second

    async def test_transients_are_not_closed_by_the_container(self, container: Container):
        # Documented contract: a transient that owns a resource leaks it.
        # Which is why anything holding a connection must be SCOPED or SINGLETON.
        async with container.scope() as scope:
            transient = await scope.get(GREETERS, "transient")

        assert not transient.closed  # type: ignore[attr-defined]


class TestLifecycle:
    async def test_components_close_in_reverse_creation_order(
        self, settings: Settings, registry: ComponentRegistry
    ):
        # A component built later may depend on one built earlier, so the later
        # one must be torn down first.
        order: list[str] = []

        class Tracked:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def greet(self) -> str:
                return self.tag

            async def aclose(self) -> None:
                order.append(self.tag)

        registry.add("llms", "first", lambda ctx: Tracked("first"))
        registry.add("llms", "second", lambda ctx: Tracked("second"))
        settings = load_settings(
            overrides={
                "llms": {
                    "default": {"provider": "first"},
                    "other": {"provider": "second"},
                }
            }
        )

        async with Container(settings, registry) as container:
            await container.get(GREETERS)
            await container.get(GREETERS, "other")

        assert order == ["second", "first"]

    async def test_every_component_is_closed_even_if_one_fails(
        self, settings: Settings, registry: ComponentRegistry
    ):
        closed: list[str] = []

        class Bad:
            def greet(self) -> str:
                return "bad"

            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        class Good:
            def greet(self) -> str:
                return "good"

            async def aclose(self) -> None:
                closed.append("good")

        registry.add("llms", "bad", lambda ctx: Bad())
        registry.add("llms", "good", lambda ctx: Good())
        settings = load_settings(
            overrides={
                "llms": {
                    "default": {"provider": "good"},
                    "other": {"provider": "bad"},
                }
            }
        )

        container = Container(settings, registry)
        await container.get(GREETERS)
        await container.get(GREETERS, "other")

        with pytest.raises(ExceptionGroup) as exc_info:
            await container.aclose()

        # The good component still closed, and the failure was not swallowed.
        assert closed == ["good"]
        assert len(exc_info.value.exceptions) == 1

    async def test_aclose_is_idempotent(self, container: Container):
        await container.get(GREETERS)
        await container.aclose()
        await container.aclose()

    async def test_a_closed_container_refuses_to_resolve(self, container: Container):
        await container.aclose()

        with pytest.raises(RegistryError, match="closed"):
            await container.get(GREETERS)


# --------------------------------------------------------------------------- #
# Bound services
# --------------------------------------------------------------------------- #


class TestBinding:
    async def test_settings_are_always_resolvable(self, container: Container):
        assert await container.get(Settings) is container.settings

    async def test_bind_an_instance(self, container: Container):
        clock = FixedClock()
        container.bind(Clock, instance=clock)

        assert await container.get(Clock) is clock

    async def test_bind_a_factory(self, container: Container):
        container.bind(Clock, factory=lambda c: FixedClock())

        assert (await container.get(Clock)).now() == 42

    async def test_a_factory_can_resolve_from_the_container(self, container: Container):
        async def build(c: Container) -> Hello:
            settings = await c.get(Settings)
            return Hello(settings.app.name)

        container.bind(Hello, factory=build)

        assert (await container.get(Hello)).greet() == "hello sarb-app"

    async def test_bound_factories_are_singletons_by_default(self, container: Container):
        container.bind(Hello, factory=lambda c: Hello())

        assert await container.get(Hello) is await container.get(Hello)

    async def test_unbound_type_is_reported(self, container: Container):
        with pytest.raises(RegistryError, match="Nothing is bound"):
            await container.get(Clock)

    async def test_binding_needs_exactly_one_of_instance_or_factory(self, container: Container):
        with pytest.raises(RegistryError, match="exactly one"):
            container.bind(Hello)

        with pytest.raises(RegistryError, match="exactly one"):
            container.bind(Hello, instance=Hello(), factory=lambda c: Hello())

    async def test_rebinding_after_resolution_is_refused(self, container: Container):
        container.bind(Hello, factory=lambda c: Hello())
        await container.get(Hello)

        # Two callers holding different objects for the same key is a bug that
        # surfaces far from its cause, so it is refused at the mistake.
        with pytest.raises(RegistryError, match="already been resolved"):
            container.bind(Hello, instance=Hello())

    async def test_a_child_scope_sees_parent_bindings(self, container: Container):
        container.bind(Hello, factory=lambda c: Hello("parent"))

        async with container.scope() as scope:
            assert (await scope.get(Hello)).greet() == "hello parent"

    async def test_a_child_binding_does_not_leak_upward(self, container: Container):
        async with container.scope() as scope:
            scope.bind(Hello, instance=Hello("child"))
            assert (await scope.get(Hello)).greet() == "hello child"

        with pytest.raises(RegistryError, match="Nothing is bound"):
            await container.get(Hello)


class TestTestOverrides:
    async def test_bind_component_replaces_a_configured_component(self, container: Container):
        fake = Hello("fake")
        container.bind_component(GREETERS, fake)

        assert await container.get(GREETERS) is fake

    async def test_an_overridden_component_needs_no_config_at_all(
        self, registry: ComponentRegistry
    ):
        # The point: a test can exercise the wiring above an LLM without config,
        # credentials, or a network.
        container = Container(load_settings(), registry)
        container.bind_component(GREETERS, Hello("fake"), name="judge")

        assert (await container.get(GREETERS, "judge")).greet() == "hello fake"

    async def test_an_overridden_component_is_not_closed_by_the_container(
        self, container: Container
    ):
        fake = Hello("fake")
        container.bind_component(GREETERS, fake)
        await container.get(GREETERS)

        await container.aclose()

        assert not fake.closed  # the test made it; the test owns it
