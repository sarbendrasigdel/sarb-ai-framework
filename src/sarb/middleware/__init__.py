"""Port decorators: retry, concurrency limits, and telemetry.

A middleware here is a class that **implements a port by wrapping another
implementation of the same port**. Because ports are ``typing.Protocol``, that works
on an adapter this framework has never seen — including one from a plugin that never
imported ``sarb``.

They are applied in exactly one place: the DI container, around whatever the registry's
factory returned. So no adapter knows they exist, no engine asks for them, and turning
retry off for one component is a line of YAML rather than a code change.

::

    Instrumented( Retrying( Limited( AnthropicProvider ) ) )
              ^         ^        ^
              |         |        └── innermost: a backing-off call holds no slot
              |         └─────────── reads .retryable, honours Retry-After
              └───────────────────── times what the *caller* waited, retries included

Why this is a ring and not a utility module
-------------------------------------------
It sits between ``contracts`` and everything that resolves a component, and it must
never learn a vendor's name — a guard that special-cased Anthropic would be a guard
that silently does nothing for the provider you write next year. The architecture test
enforces that: ``middleware`` may not import ``providers``, ``engines``, ``api`` or
``cli``.
"""

from __future__ import annotations

from typing import Any

from sarb.config import ComponentConfig, ResilienceSettings, Settings
from sarb.contracts import EmbeddingProvider, LLMProvider, VectorStore
from sarb.middleware.embedding import GuardedEmbedder
from sarb.middleware.guard import Guard, Outcome
from sarb.middleware.llm import GuardedLLM
from sarb.middleware.vectorstore import GuardedVectorStore

__all__ = [
    "Guard",
    "GuardedEmbedder",
    "GuardedLLM",
    "GuardedVectorStore",
    "Outcome",
    "policy_for",
    "wrap",
]

#: Port -> the wrapper that decorates it.
#:
#: Keyed on the **port**, not on the family name, and the distinction is not academic.
#: A family name is a config key; a port is a contract. ``Family("llms", Greeter)`` is a
#: perfectly legal thing for a plugin — or a test — to construct, and a table keyed on
#: ``"llms"`` would cheerfully wrap that ``Greeter`` in a ``GuardedLLM`` and hand back
#: something with no ``greet`` on it. A decorator decorates a *port*. That is the only
#: thing it can safely be selected by.
#:
#: A port with no entry here is returned unwrapped, deliberately. A plugin that adds a
#: reranker gets no retry until someone writes ``GuardedReranker`` — and quietly
#: returning the bare component is the honest outcome, because the alternative is a
#: ``__getattr__`` proxy that would make ``isinstance(x, AnythingAtAll)`` answer
#: ``True`` and defeat the type checker on the way past.
_WRAPPERS: dict[Any, Any] = {
    LLMProvider: GuardedLLM,
    EmbeddingProvider: GuardedEmbedder,
    VectorStore: GuardedVectorStore,
}


def policy_for(config: ComponentConfig, settings: Settings) -> ResilienceSettings:
    """The resilience policy in force for one component.

    A component's own ``resilience:`` block wins outright; otherwise it inherits the
    application-wide default. Deliberately *not* a field-by-field merge — a partial
    override reads as "these two settings, plus whatever the default happens to be",
    which is a sentence nobody can act on when the default later changes underneath it.
    """
    return config.resilience or settings.resilience


def wrap(
    component: Any,
    *,
    port: Any,
    path: str,
    config: ComponentConfig,
    settings: Settings,
) -> Any:
    """Put a guard around a freshly built component.

    Args:
        component: What the adapter's factory returned.
        port: The port it was built to satisfy, e.g. ``LLMProvider``.
        path: ``"llms.judge"`` — carried on every span, log line and metric.
        config: The component's config block.
        settings: The application settings, for the inherited policy.

    Returns:
        The component, wrapped — or the component itself, for a port with no wrapper.
    """
    wrapper = _WRAPPERS.get(port)
    if wrapper is None:
        return component

    guard = Guard(
        component=path,
        provider=config.provider,
        policy=policy_for(config, settings),
        pricing=settings.pricing,
    )
    return wrapper(component, guard)
