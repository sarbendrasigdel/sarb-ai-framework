"""Ambient fields that every log record and span inherits.

The problem this solves: an HTTP request enters, fans out through a retrieval
engine, three tool calls and two model calls, and produces forty log lines across
a dozen modules. Every one of those lines needs the ``request_id``, or none of
them can be reassembled into the story of what happened to one user.

Passing a correlation id through forty call signatures is not a design; it is a
tax that gets skipped the moment somebody is in a hurry. So the id lives in a
:class:`~contextvars.ContextVar`, which asyncio propagates into every task spawned
inside the block:

    with log_context(request_id=req_id, user_id=user.id):
        await handle(request)          # every record inside carries both

Context is *additive and nested*: an inner block adds fields without erasing the
outer ones, and restores exactly the previous state on exit — including when the
block raises.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

__all__ = ["current_context", "log_context"]

#: The default is a read-only mapping, and every update *replaces* it rather than
#: mutating in place. This is not fussiness: a plain dict stored in a ContextVar is
#: shared by reference with every task that inherited the context, so mutating it
#: would be visible to sibling tasks — a cross-request data leak wearing a very
#: ordinary-looking disguise.
_EMPTY: Mapping[str, Any] = MappingProxyType({})

_context: ContextVar[Mapping[str, Any]] = ContextVar("sarb_log_context", default=_EMPTY)


def current_context() -> Mapping[str, Any]:
    """The fields currently bound to this task.

    Returns:
        An immutable view of the ambient fields. Empty when nothing is bound.
    """
    return _context.get()


@contextmanager
def log_context(**fields: Any) -> Iterator[Mapping[str, Any]]:
    """Bind fields for the enclosing block, and everything it awaits.

    Args:
        **fields: Values to attach to every record and span created inside.

    Yields:
        The merged context, outer fields plus these.

    Example:
        >>> with log_context(request_id="abc"):
        ...     with log_context(agent="researcher"):
        ...         logger.info("step")   # carries request_id AND agent
        ...     logger.info("done")       # carries request_id only
    """
    merged = MappingProxyType({**_context.get(), **fields})
    token = _context.set(merged)
    try:
        yield merged
    finally:
        _context.reset(token)
