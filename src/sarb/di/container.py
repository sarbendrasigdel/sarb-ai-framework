"""The dependency-injection container: the one place the rings are joined.

Every other module in this framework is written against ports and knows nothing
about the adapters behind them. That ignorance is only affordable because
*something* eventually decides that ``llms.default`` means an ``AnthropicProvider``
holding a real HTTP pool. This is that something, and it is deliberately the only
one — a module that constructs its own provider has quietly moved the composition
root into itself and taken its own replaceability with it.

The container resolves two kinds of thing:

**Configured components**, by family and name. What to build comes from settings,
how to build it from the registry::

    llm = await container.get(LLMS)            # settings.llms["default"]
    judge = await container.get(LLMS, "judge") # settings.llms["judge"]

**Bound services**, by type. Anything the composition root wires by hand::

    container.bind(Clock, factory=lambda c: SystemClock())
    clock = await container.get(Clock)

Design decisions worth defending
--------------------------------
**No auto-wiring.** The container will not introspect ``__init__`` annotations and
magically construct dependencies. Auto-wiring is delightful in a demo and awful in
year three: it fails at runtime, in a stack trace nobody can read, usually during
a deploy. Explicit factories cost three lines and stay debuggable forever.

**Request scope is not silently promoted.** Resolving a ``SCOPED`` component from
the root raises. The alternative — quietly handing back a singleton — is the bug
where one user's request-bound session is captured by a long-lived object and
served to the next user.

**Shutdown is LIFO, and never swallows.** Components close in reverse creation
order, because a component built later may depend on one built earlier. Every
close is attempted even if an earlier one fails, and the failures surface as an
``ExceptionGroup`` rather than being logged and forgotten.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Self, cast, overload

import anyio

from sarb.config import Settings
from sarb.contracts.component import (
    DEFAULT_COMPONENT,
    Closable,
    Family,
    Scope,
)
from sarb.errors import ConfigurationError, RegistryError, SarbError
from sarb.middleware import wrap as _wrap_component
from sarb.plugins.registry import ComponentContext, ComponentRegistry

__all__ = ["Container", "ServiceFactory"]

#: Builds a bound service. Receives the container so it can resolve what it needs.
#: Split across two callables for the same reason as
#: :data:`~sarb.plugins.registry.ComponentFactory`: the single-callable form makes
#: ``T`` uninferable for ``async def`` factories.
type ServiceFactory[T] = Callable[["Container"], T] | Callable[["Container"], Awaitable[T]]

#: Cache key: a type for bound services, a triple for configured components.
type _Key = type[Any] | tuple[str, str, str]

#: The chain of components currently being constructed on this task, used to turn
#: a circular dependency into a readable error instead of a RecursionError or —
#: worse — a deadlock on the per-key locks below.
_resolving: ContextVar[tuple[str, ...]] = ContextVar("sarb_di_resolving", default=())


@dataclass(frozen=True, slots=True)
class _Binding:
    """A service bound by type, as opposed to a component configured by name."""

    scope: Scope
    factory: ServiceFactory[Any] | None = None
    instance: Any = None


class Container:
    """Owns component instances and their lifecycles.

    The root container holds singletons and lives as long as the application.
    Child scopes — one per request, typically — hold scoped instances and are
    closed when the request ends.

    Example:
        ```python
        settings = load_settings(config_dir="config", profile="production")
        registry = build_registry(settings)

        async with Container(settings, registry) as container:
            llm = await container.get(LLMS)              # singleton, cached

            async with container.scope() as request:
                memory = await request.get(MEMORIES)     # per-request
            # `memory` is closed here; `llm` is not
        # `llm` is closed here
        ```
    """

    def __init__(
        self,
        settings: Settings,
        registry: ComponentRegistry,
        *,
        parent: Container | None = None,
    ) -> None:
        """Create a container.

        Args:
            settings: Application settings. Decides *what* to build.
            registry: Component registry. Decides *how* to build it.
            parent: Set internally by :meth:`scope`. Application code builds the
                root container and calls :meth:`scope` for children.
        """
        self._settings = settings
        self._registry = registry
        self._parent = parent

        self._bindings: dict[type[Any], _Binding] = {}
        self._instances: dict[_Key, Any] = {}
        self._locks: dict[_Key, anyio.Lock] = {}
        self._closables: list[Closable] = []
        self._closed = False

        if parent is None:
            # Settings are always resolvable; every other binding is the
            # composition root's business.
            self._bindings[Settings] = _Binding(scope=Scope.SINGLETON, instance=settings)

    # -- properties --------------------------------------------------------- #

    @property
    def settings(self) -> Settings:
        """The application settings this container resolves against."""
        return self._settings

    @property
    def registry(self) -> ComponentRegistry:
        """The registry this container resolves providers from."""
        return self._registry

    @property
    def is_root(self) -> bool:
        """Whether this is the application container rather than a child scope."""
        return self._parent is None

    def _root(self) -> Container:
        """The container that owns singletons."""
        node = self
        while node._parent is not None:
            node = node._parent
        return node

    # -- binding ------------------------------------------------------------ #

    def bind[T](
        self,
        key: type[T],
        *,
        instance: T | None = None,
        factory: ServiceFactory[T] | None = None,
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        """Bind a service to a type, by instance or by factory.

        Args:
            key: The type callers will resolve, usually a Protocol.
            instance: A ready-made object. Implies singleton scope; the container
                will still close it at shutdown if it is closable.
            factory: Builds the service. Receives the container, so it can
                resolve its own dependencies. May be sync or async.
            scope: Lifetime. Ignored when ``instance`` is given.

        Raises:
            RegistryError: If neither or both of ``instance`` and ``factory`` are
                given, or if ``key`` has already been resolved. Rebinding
                something already handed out means two callers hold different
                objects for the same key — a bug that shows up far from its cause,
                so it is refused at the point of the mistake.
        """
        if (instance is None) == (factory is None):
            raise RegistryError(
                f"Binding {key.__name__} needs exactly one of instance= or factory=.",
                context={"key": key.__name__},
            )

        if key in self._instances:
            raise RegistryError(
                f"{key.__name__} has already been resolved and cannot be rebound. "
                "Bind everything before the first resolve, or use a child scope.",
                context={"key": key.__name__},
            )

        self._bindings[key] = _Binding(
            scope=Scope.SINGLETON if instance is not None else scope,
            factory=factory,
            instance=instance,
        )

    def bind_component[PortT](
        self,
        family: Family[PortT],
        instance: PortT,
        *,
        name: str = DEFAULT_COMPONENT,
    ) -> None:
        """Substitute a ready-made object for a configured component.

        The seam tests use: swap a fake LLM into the container and exercise the
        real wiring above it, without config, credentials, or a network.

        The substituted object is **not** closed by the container — whoever
        created it owns it.

        Args:
            family: The family to override, e.g. ``LLMS``.
            instance: The object to return instead of building one.
            name: The component name to override.

        Example:
            >>> container.bind_component(LLMS, FakeLLM(), name="judge")
        """
        self._instances[self._component_key(family.name, name)] = instance

    # -- resolution --------------------------------------------------------- #

    @overload
    async def get[T](self, key: type[T], /) -> T: ...

    @overload
    async def get[PortT](self, key: Family[PortT], /, name: str = DEFAULT_COMPONENT) -> PortT: ...

    async def get(
        self,
        key: type[Any] | Family[Any],
        /,
        name: str = DEFAULT_COMPONENT,
    ) -> Any:
        """Resolve a bound service by type, or a configured component by family.

        Args:
            key: A type bound with :meth:`bind`, or a
                :class:`~sarb.contracts.component.Family`.
            name: Component name; only meaningful when ``key`` is a family.

        Returns:
            The resolved instance, typed as the requested type or the family's port.

        Raises:
            RegistryError: If the type is not bound, or a dependency cycle exists.
            ConfigurationError: If the component is not configured or is disabled.
            ProviderNotFoundError: If config names a provider nothing registered.
        """
        if isinstance(key, Family):
            return await self.resolve(key, name)
        return await self._get_service(key)

    async def resolve[PortT](
        self,
        family: Family[PortT],
        name: str = DEFAULT_COMPONENT,
    ) -> PortT:
        """Resolve one configured component. Satisfies ``ComponentResolver``.

        Args:
            family: The family to resolve from, e.g. ``LLMS``.
            name: The component name within the family.

        Returns:
            The component, typed as the family's port.

        Raises:
            ConfigurationError: If the component is absent from settings, or is
                explicitly disabled.
            ProviderNotFoundError: If its ``provider`` key is not registered.
        """
        self._check_open()

        cache_key = self._component_key(family.name, name)

        # Test overrides and already-built instances short-circuit before any
        # config is consulted, so a bound fake needs no config block at all.
        found = self._lookup_instance(cache_key)
        if found is not _MISSING:
            return cast("PortT", found)

        config = self._settings.component(family.name, name)
        path = f"{family.name}.{name}"

        if not config.enabled:
            raise ConfigurationError(
                f"Component {path!r} is disabled. Set enabled: true to use it.",
                context={"family": family.name, "name": name, "provider": config.provider},
            )

        registration = self._registry.lookup(family.name, config.provider)

        async def build() -> Any:
            context = ComponentContext(
                family=family.name,
                name=name,
                config=config,
                settings=self._settings,
                resolver=self,
            )
            component = registration.factory(context)

            # A factory may be async — a connection pool cannot be opened without an
            # event loop, so `pgvector`'s is. Its result must be awaited *before* the
            # wrapping below, or the guard would be handed a coroutine and would
            # faithfully wrap that instead of the component inside it.
            if inspect.isawaitable(component):
                component = await component

            # The composition root is the only honest place for cross-cutting policy.
            # Retry, concurrency limits and telemetry are applied here, around whatever
            # the factory returned — so no adapter has to remember them, no engine has
            # to ask for them, and a provider written by a plugin next year gets them
            # without having heard of them. See `sarb.middleware`.
            return _wrap_component(
                component,
                port=family.port,
                path=path,
                config=config,
                settings=self._settings,
            )

        instance = await self._get_or_create(
            cache_key,
            registration.scope,
            build,
            label=f"{path} ({config.provider})",
        )
        # Verified *after* wrapping, on purpose: the wrapper is what callers receive, so
        # the wrapper is what has to satisfy the port. A guard that quietly dropped a
        # method would otherwise be caught by nothing until something called it.
        self._verify_port(instance, family, path, config.provider)
        return cast("PortT", instance)

    async def _get_service[T](self, key: type[T]) -> T:
        """Resolve a service bound by type."""
        self._check_open()

        found = self._lookup_instance(key)
        if found is not _MISSING:
            return cast("T", found)

        binding = self._find_binding(key)
        if binding is None:
            raise RegistryError(
                f"Nothing is bound for {key.__name__}. Bind it on the container "
                "before resolving: container.bind(Type, factory=...).",
                context={"key": key.__name__},
            )

        if binding.instance is not None:
            return cast("T", binding.instance)

        factory = binding.factory
        assert factory is not None  # guaranteed by bind()'s exactly-one check

        instance = await self._get_or_create(
            key,
            binding.scope,
            lambda: factory(self),
            label=key.__name__,
        )
        return cast("T", instance)

    def _find_binding(self, key: type[Any]) -> _Binding | None:
        """Find a binding here or on any ancestor."""
        node: Container | None = self
        while node is not None:
            binding = node._bindings.get(key)
            if binding is not None:
                return binding
            node = node._parent
        return None

    def _lookup_instance(self, cache_key: _Key) -> Any:
        """Find an existing instance here or on any ancestor."""
        node: Container | None = self
        while node is not None:
            if cache_key in node._instances:
                return node._instances[cache_key]
            node = node._parent
        return _MISSING

    # -- construction ------------------------------------------------------- #

    async def _get_or_create(
        self,
        cache_key: _Key,
        scope: Scope,
        build: Callable[[], Any],
        *,
        label: str,
    ) -> Any:
        """Build an instance in the container that owns its scope, or reuse it."""
        if scope is Scope.TRANSIENT:
            # Deliberately uncached and untracked. See Scope.TRANSIENT.
            self._guard_cycle(label)
            return await self._instantiate(build, label)

        owner = self._owner_for(scope, label)

        if cache_key in owner._instances:
            return owner._instances[cache_key]

        # Cycles must be caught *before* the lock. A -> B -> A would otherwise
        # re-enter this method for A while A's lock is still held by the task
        # that is building it, and deadlock on a non-reentrant lock instead of
        # reporting the cycle.
        self._guard_cycle(label)

        # One lock per key, not one per container: two different singletons can
        # be built concurrently, while ten tasks racing for the *same* singleton
        # produce exactly one instance.
        lock = owner._locks.setdefault(cache_key, anyio.Lock())
        async with lock:
            if cache_key in owner._instances:
                return owner._instances[cache_key]

            instance = await self._instantiate(build, label)

            owner._instances[cache_key] = instance
            if isinstance(instance, Closable):
                owner._closables.append(instance)
            return instance

    @staticmethod
    def _guard_cycle(label: str) -> None:
        """Refuse to re-enter a component that is already under construction."""
        chain = _resolving.get()
        if label in chain:
            cycle = " -> ".join([*chain, label])
            raise RegistryError(
                f"Circular dependency: {cycle}",
                context={"cycle": [*chain, label]},
            )

    def _owner_for(self, scope: Scope, label: str) -> Container:
        """Which container owns instances of this scope."""
        if scope is Scope.SINGLETON:
            return self._root()

        if self.is_root:
            raise RegistryError(
                f"{label} is request-scoped and cannot be resolved from the "
                "application container. Open a scope first: "
                "`async with container.scope() as request:`",
                context={"component": label, "scope": scope.value},
            )
        return self

    async def _instantiate(self, build: Callable[[], Any], label: str) -> Any:
        """Run a factory, recording it on the resolution chain and awaiting it.

        The chain is what makes a cycle legible; :meth:`_guard_cycle` reads it
        before any lock is taken.
        """
        token = _resolving.set((*_resolving.get(), label))
        try:
            result = build()
            if inspect.isawaitable(result):
                result = await result
        except SarbError:
            # Already actionable — a ConfigurationError from parse_options names
            # the exact YAML key. Wrapping it would only bury the message.
            raise
        except Exception as exc:
            raise RegistryError(
                f"Failed to build {label}: {exc}",
                context={"component": label},
                cause=exc,
            ) from exc
        finally:
            _resolving.reset(token)

        return result

    @staticmethod
    def _verify_port(instance: Any, family: Family[Any], path: str, provider: str) -> None:
        """Check a freshly built component actually satisfies its port.

        Only possible for ``runtime_checkable`` protocols, and even then this only
        confirms the methods exist — not their signatures. That is still worth
        doing: it catches an adapter that has drifted out of shape at wiring time,
        rather than three layers deep on the first call in production.
        """
        if not getattr(family.port, "_is_runtime_protocol", False):
            return
        if not isinstance(instance, family.port):
            raise RegistryError(
                f"Provider {provider!r} built for {path!r} does not satisfy "
                f"{family.port.__name__}: {type(instance).__name__} is missing "
                "one or more of its methods.",
                context={"component": path, "provider": provider, "port": family.port.__name__},
            )

    @staticmethod
    def _component_key(family: str, name: str) -> tuple[str, str, str]:
        """Cache key for a configured component."""
        return ("component", family, name)

    # -- lifecycle ---------------------------------------------------------- #

    @asynccontextmanager
    async def scope(self) -> AsyncIterator[Container]:
        """Open a child scope, e.g. for one HTTP request.

        Scoped components resolved inside are created here and closed on exit.
        Singletons still come from the root and are untouched.

        Yields:
            A child container.

        Example:
            >>> async with container.scope() as request:
            ...     memory = await request.get(MEMORIES)
        """
        self._check_open()
        child = Container(self._settings, self._registry, parent=self)
        try:
            yield child
        finally:
            await child.aclose()

    async def aclose(self) -> None:
        """Close every component this container owns, newest first.

        Idempotent. Every component is closed even if an earlier one raises;
        the failures are collected and raised together, because a shutdown that
        silently drops errors is how connection leaks survive to production.

        Raises:
            ExceptionGroup: If any component raised while closing.
        """
        if self._closed:
            return
        self._closed = True

        failures: list[Exception] = []
        for closable in reversed(self._closables):
            try:
                await closable.aclose()
            except Exception as exc:
                # One bad close must not prevent the rest from running.
                failures.append(exc)

        self._closables.clear()
        self._instances.clear()
        self._locks.clear()

        if failures:
            raise ExceptionGroup(f"{len(failures)} component(s) failed to close cleanly", failures)

    def _check_open(self) -> None:
        """Refuse to serve a container that has already been shut down."""
        if self._closed:
            raise RegistryError("This container is closed and can no longer resolve.")

    async def __aenter__(self) -> Self:
        """Enter the container's lifetime."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close everything the container owns."""
        await self.aclose()

    def __repr__(self) -> str:
        """Summarise what this container currently holds."""
        kind = "root" if self.is_root else "scope"
        return f"Container({kind}, instances={len(self._instances)})"


class _Missing:
    """Sentinel distinguishing "not cached" from a legitimately cached ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()
