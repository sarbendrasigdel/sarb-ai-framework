"""The vocabulary of wiring: families, scopes, lifecycle.

These are the few types that the configuration layer, the plugin registry and
the DI container must all agree on. They live in ``contracts`` — the innermost
ring — precisely because all three of those modules depend on them, and none of
them may depend on each other.

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_COMPONENT",
    "Closable",
    "ComponentResolver",
    "Family",
    "Scope",
]

#: The component name used when a caller does not ask for one explicitly.
DEFAULT_COMPONENT = "default"


class Scope(StrEnum):
    """How long a resolved instance lives, and who closes it.

    Attributes:
        SINGLETON: One instance per application, created on first use and closed
            at shutdown. The right choice for anything holding a connection pool
            — an LLM client, a vector store, a database.
        SCOPED: One instance per scope, typically per HTTP request. Closed when
            the scope exits. The right choice for anything carrying per-request
            state: a unit of work, a conversation buffer, a request-bound
            database session.
        TRANSIENT: A fresh instance every time, **never tracked and never closed
            by the container**. Only for cheap, resource-free objects. A
            transient that owns a socket will leak it; make it SCOPED instead.
    """

    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


@runtime_checkable
class Closable(Protocol):
    """Something the container must shut down.

    Any component with an ``aclose`` method is closed automatically when its
    owning scope exits — in reverse order of creation, since a component built
    later may depend on one built earlier.

    Structural, not inherited: a vendor's own client class that happens to have
    ``aclose`` already satisfies this, with no adapter shim.
    """

    async def aclose(self) -> None:
        """Release the resources this component owns. Must be idempotent."""
        ...


@dataclass(frozen=True, slots=True)
class Family[PortT]:
    """Binds a configuration family to the port its members implement.

    A family is the link between three things that would otherwise have to know
    about each other: the config key operators write (``llms:``), the port
    engines depend on (``LLMProvider``), and the adapters registered against it
    (``anthropic``, ``openai``).

    Because it is generic in the port, resolution is type-safe end to end::

        LLMS = Family("llms", LLMProvider)

        llm = await container.get(LLMS)            # inferred: LLMProvider
        judge = await container.get(LLMS, "judge")

    Attributes:
        name: The settings attribute holding this family, e.g. ``"llms"`` for
            ``settings.llms``. Also the registry namespace.
        port: The Protocol every member of the family must satisfy. When the
            port is ``runtime_checkable``, the container verifies newly built
            components against it — so an adapter that drifts out of shape is
            caught at wiring time rather than at the first call.
    """

    name: str
    port: type[PortT]

    def __str__(self) -> str:
        """Render as the config key, which is how operators refer to it."""
        return self.name


@runtime_checkable
class ComponentResolver(Protocol):
    """The slice of the container that a component factory is allowed to see.

    A factory building a retriever needs to reach an LLM and a vector store. It
    does **not** need — and must not have — the container's binding, scoping and
    shutdown machinery. Handing factories this narrow protocol instead of the
    container itself keeps the blast radius of a plugin small, and keeps
    ``plugins`` free of any dependency on ``di``.
    """

    async def resolve[PortT](
        self,
        family: Family[PortT],
        name: str = DEFAULT_COMPONENT,
    ) -> PortT:
        """Resolve one configured component by family and name."""
        ...
