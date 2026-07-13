"""Adapters: the only ring in the framework that is allowed to name a vendor.

Everything under this package implements a port from :mod:`sarb.contracts` on top
of exactly one external system. Nothing above it — no engine, no router, no CLI
command — may import from here. That rule is enforced by
``tests/architecture/test_dependency_rule.py``, and it is what makes a vendor swap
a config change rather than a project.

Two conventions hold for every adapter, without exception.

**The vendor SDK is imported lazily, inside the factory.** Never at module level.
Importing ``sarb.providers.llm.anthropic`` on a core-only install must succeed and
must be cheap; only *building* the component may fail, and when it does it fails
with ``uv add "sarb[anthropic]"`` rather than a ``ModuleNotFoundError`` from four
frames down. See :func:`sarb.dependencies.import_optional`.

**The vendor client is injected, not constructed.** An adapter's ``__init__``
takes an already-built client; its ``build(ctx)`` factory is what reaches for the
SDK and the API key. The split is not ceremony — it is what lets the translation
layer, which is where all the real risk lives, be tested against the vendor's own
response models without a network call or a key.
"""

from __future__ import annotations

__all__: list[str] = []
