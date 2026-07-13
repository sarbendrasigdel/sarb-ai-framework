"""Dependency injection: the only place the rings are joined.

    async with bootstrap(config_dir="config", profile="production") as container:
        llm = await container.get(LLMS)              # from settings.llms.default

        async with container.scope() as request:
            memory = await request.get(MEMORIES)     # closed when the scope exits

Configured components are resolved by :class:`~sarb.contracts.component.Family`;
services wired by hand are bound and resolved by type. See
:mod:`sarb.di.container` for the design decisions — in particular why there is no
auto-wiring, and why a request-scoped component cannot be resolved from the root.
"""

from sarb.contracts.component import Family, Scope
from sarb.di.bootstrap import bootstrap, build_container
from sarb.di.container import Container, ServiceFactory

__all__ = [
    "Container",
    "Family",
    "Scope",
    "ServiceFactory",
    "bootstrap",
    "build_container",
]
