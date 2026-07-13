"""The component registry: which adapter answers to which config key.

The registry is a lookup table from ``(family, provider_key)`` to a **factory**.
It holds no instances and opens no connections; that is the container's job.

Two properties of the design carry most of the weight.

**Registration does not import the vendor SDK.** A registration records a
callable, and the ``import anthropic`` happens inside that callable, the first
time a component is actually built. So every first-party adapter is registered
unconditionally, even when its extra is not installed — and asking for a
provider you have not installed produces

    Provider 'qdrant' requires an optional dependency: pip install 'sarb[qdrant]'

instead of an ``ImportError`` from somewhere in the import graph. This is the
whole reason ``pip install sarb`` can stay small without making the missing-extra
experience miserable.

**Plugins are handed the registry; they never mutate a global.** A plugin's entry
point resolves to ``register(registry)``. There is no import-time global to
corrupt, no ordering dependency between plugins, and a test can build a
throwaway registry that vanishes with the test.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sarb.config import ComponentConfig, Settings
from sarb.contracts.component import ComponentResolver, Family, Scope
from sarb.errors import ProviderNotFoundError, RegistryError

__all__ = [
    "ComponentContext",
    "ComponentFactory",
    "ComponentRegistry",
    "PluginRegistrar",
    "Registration",
]


@dataclass(frozen=True, slots=True)
class ComponentContext:
    """Everything a factory needs, and nothing it does not.

    Attributes:
        family: The family being built, e.g. ``"llms"``.
        name: The component's name within the family, e.g. ``"judge"``.
        config: The component's own config block — ``provider`` plus the opaque
            ``options`` this factory is responsible for validating.
        settings: The full application settings. For cross-cutting values only
            (timeouts, environment); a factory reading *another component's*
            options has coupled itself to that component's schema.
        resolver: For resolving dependencies on other components — a retriever
            factory reaching for its vector store. Deliberately the narrow
            :class:`~sarb.contracts.component.ComponentResolver` protocol rather
            than the container itself, so a factory cannot rebind or shut down
            the application it lives in.
    """

    family: str
    name: str
    config: ComponentConfig
    settings: Settings
    resolver: ComponentResolver

    @property
    def path(self) -> str:
        """Dotted config path, e.g. ``"llms.judge"``.

        Pass this to :func:`sarb.config.parse_options` so option errors name the
        exact YAML key an operator has to change.
        """
        return f"{self.family}.{self.name}"


#: Builds one component. May be sync or async — a factory that has to open a
#: connection should be async rather than blocking the event loop in ``__init__``.
#:
#: Written as a union of two callables rather than the tidier
#: ``Callable[[ComponentContext], T | Awaitable[T]]``, because that form gives
#: type checkers nothing to infer ``T`` from when the factory is ``async def``:
#: they solve it to ``Never`` and every registration fails to type check.
type ComponentFactory[T] = (
    Callable[[ComponentContext], T] | Callable[[ComponentContext], Awaitable[T]]
)

#: What a plugin's entry point must resolve to. Called once at startup with the
#: registry to populate. Anything else raises :class:`~sarb.errors.PluginError`.
type PluginRegistrar = Callable[[ComponentRegistry], None]


@dataclass(frozen=True, slots=True)
class Registration:
    """One adapter, registered against one family.

    Attributes:
        family: The family it belongs to, e.g. ``"vector_stores"``.
        key: The value operators write as ``provider:`` in config, e.g. ``"qdrant"``.
        factory: Builds the component. Not called until the component is needed.
        scope: Lifetime of the built instance. Defaults to
            :attr:`~sarb.contracts.component.Scope.SINGLETON`, which is correct
            for anything holding a connection pool.
        summary: One line, shown by ``sarb providers list``. Written for the
            operator choosing between adapters, not for the developer.
        source: Where the registration came from — ``"builtin"`` or a plugin
            name. Makes "why is this adapter here?" answerable at runtime.
    """

    family: str
    key: str
    factory: ComponentFactory[Any]
    scope: Scope = Scope.SINGLETON
    summary: str = ""
    source: str = "builtin"

    @property
    def path(self) -> str:
        """Dotted identity, e.g. ``"vector_stores.qdrant"``."""
        return f"{self.family}.{self.key}"


class ComponentRegistry:
    """A mutable map of ``(family, provider_key)`` to :class:`Registration`.

    Built once at startup (see :func:`sarb.plugins.build_registry`), then read.

    Example:
        >>> registry = ComponentRegistry()
        >>> registry.add("llms", "anthropic", build_anthropic, summary="Claude models")
        >>> registry.lookup("llms", "anthropic").scope
        <Scope.SINGLETON: 'singleton'>
    """

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], Registration] = {}

    def add(
        self,
        family: str,
        key: str,
        factory: ComponentFactory[Any],
        *,
        scope: Scope = Scope.SINGLETON,
        summary: str = "",
        source: str = "builtin",
        replace: bool = False,
    ) -> None:
        """Register an adapter under a family and provider key.

        Deliberately not generic in the component type. The registry is a
        heterogeneous map — one table holding LLMs, vector stores and memories —
        so a type variable here would have nothing to bind against and buys no
        safety. Type safety is recovered where it is actually useful: at
        :meth:`sarb.di.Container.resolve`, which is generic in the
        :class:`~sarb.contracts.component.Family`'s port and additionally checks
        the built instance against it.

        Args:
            family: Config family, e.g. ``"llms"``.
            key: Provider key operators write in config, e.g. ``"anthropic"``.
            factory: Builds the component when it is first needed. Must not
                import its vendor SDK until called.
            scope: Lifetime of the built instance.
            summary: One-line description for humans.
            source: ``"builtin"``, or the name of the plugin registering it.
            replace: Permit overwriting an existing registration. Required, and
                deliberately explicit — see Raises.

        Raises:
            RegistryError: If ``(family, key)`` is already registered and
                ``replace`` is false. Two plugins silently claiming ``"qdrant"``
                would make behaviour depend on installation order, which is the
                kind of bug that takes a week to find. Overriding is legitimate,
                so it is allowed — but it has to be said out loud.
        """
        identity = (family, key)
        existing = self._registrations.get(identity)
        if existing is not None and not replace:
            raise RegistryError(
                f"{family}.{key} is already registered by {existing.source!r}. "
                f"Pass replace=True to override it deliberately.",
                context={
                    "family": family,
                    "key": key,
                    "existing_source": existing.source,
                    "new_source": source,
                },
            )

        self._registrations[identity] = Registration(
            family=family,
            key=key,
            factory=factory,
            scope=scope,
            summary=summary,
            source=source,
        )

    def lookup(self, family: str, key: str) -> Registration:
        """Find the adapter registered for a family and provider key.

        Args:
            family: Config family, e.g. ``"vector_stores"``.
            key: Provider key from config, e.g. ``"qdrant"``.

        Returns:
            The matching registration.

        Raises:
            ProviderNotFoundError: If nothing is registered. The message lists
                every provider that *is* available for the family, because the
                cause is nearly always a typo, a missing extra, or a plugin that
                failed to load — and all three are answered by that list.
        """
        try:
            return self._registrations[(family, key)]
        except KeyError:
            available = self.providers(family)
            hint = ", ".join(available) if available else "<none registered>"
            raise ProviderNotFoundError(
                f"No provider {key!r} registered for family {family!r}. Available: {hint}",
                context={"family": family, "key": key, "available": list(available)},
            ) from None

    def providers(self, family: str) -> tuple[str, ...]:
        """Every provider key registered for a family, sorted."""
        return tuple(sorted(k for f, k in self._registrations if f == family))

    def families(self) -> tuple[str, ...]:
        """Every family that has at least one registered provider, sorted."""
        return tuple(sorted({f for f, _ in self._registrations}))

    def registrations(self) -> tuple[Registration, ...]:
        """Every registration, sorted by family then key. For the CLI and docs."""
        return tuple(sorted(self._registrations.values(), key=lambda r: (r.family, r.key)))

    def supports(self, family: Family[Any] | str, key: str) -> bool:
        """Whether a provider key is registered for a family."""
        name = family if isinstance(family, str) else family.name
        return (name, key) in self._registrations

    def __len__(self) -> int:
        """Number of registered adapters."""
        return len(self._registrations)

    def __repr__(self) -> str:
        """Summarise the registry by family."""
        by_family = {f: len(self.providers(f)) for f in self.families()}
        return f"ComponentRegistry({by_family})"
