"""sarb — a modular framework for building production AI agent applications.

The framework is organised in rings, and dependencies point inward only:

``sarb.contracts``
    Ports (``typing.Protocol``) and domain models. Depends on nothing but
    Pydantic and the standard library.
``sarb.engines``
    Use-cases: agents, retrieval, memory, workflows, documents, tools. Depends on
    ``contracts`` only — an engine never learns that OpenAI or Qdrant exist.
``sarb.providers``
    Adapters implementing the ports against real vendors. Depends on
    ``contracts`` plus one vendor SDK, imported lazily behind an extra.
``sarb.api`` / ``sarb.cli``
    Delivery. Depends on engines.
``sarb.config`` / ``sarb.di`` / ``sarb.observability`` / ``sarb.plugins``
    The cross-cutting kernel, and the only place where the rings are wired
    together.

The rule that makes the rest work: **engines must never import providers.**
Wiring happens once, from configuration, in the DI container. It is enforced by
an automated test, not by convention.
"""

from sarb.errors import SarbError

__version__ = "0.1.0"

__all__ = ["SarbError", "__version__"]
