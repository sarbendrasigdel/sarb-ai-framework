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

from importlib.metadata import PackageNotFoundError, version

from sarb.errors import SarbError

try:
    __version__ = version("sarb")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    # Not installed — a source tree being read directly, which is what the test suite and
    # `python -m` do. There is no distribution to ask, and inventing a number here would
    # be a lie with a version attached.
    __version__ = "0.0.0+unknown"

#: Read from the installed distribution rather than written here, because a version
#: written in two places is a version that will be bumped in one of them. `pyproject.toml`
#: is the source of truth; this asks it what it said.
__all__ = ["SarbError", "__version__"]
