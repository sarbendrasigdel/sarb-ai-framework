"""Importing an optional dependency, and failing usefully when it is absent.

The framework's core is small on purpose: vendor SDKs and heavy runtimes live
behind extras. The cost of that choice is paid at exactly one moment — when
somebody configures a provider whose extra they have not installed — and this
module exists to make that moment cheap.

Without it::

    ModuleNotFoundError: No module named 'qdrant_client'

With it::

    Provider 'qdrant' needs an optional dependency that is not installed.
    Install it with:  uv add "sarb[qdrant]"

The difference is a developer reading a traceback and guessing, versus a developer
running one command. Every adapter therefore imports its SDK through
:func:`import_optional`, inside its factory — never at module level.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from sarb.errors import MissingDependencyError

__all__ = ["import_optional"]


def import_optional(module: str, *, extra: str, used_by: str = "") -> ModuleType:
    """Import a module that lives behind an optional extra.

    Args:
        module: The import name, e.g. ``"qdrant_client"``.
        extra: The extra that provides it, e.g. ``"qdrant"`` — the name that goes
            inside the brackets of ``sarb[...]``.
        used_by: What needed it, e.g. ``"provider 'qdrant'"``. Appears in the
            error, because the operator's next question is "needed by what?".

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module is not installed. The message names
            the exact command to run.

    Example:
        >>> def build(ctx: ComponentContext) -> QdrantStore:
        ...     qdrant = import_optional(
        ...         "qdrant_client", extra="qdrant", used_by="provider 'qdrant'"
        ...     )
        ...     return QdrantStore(qdrant.AsyncQdrantClient(...))
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        who = f"{used_by} needs" if used_by else f"{module!r} is"
        raise MissingDependencyError(
            f"{who} an optional dependency that is not installed. "
            f'Install it with:  uv add "sarb[{extra}]"',
            context={"module": module, "extra": extra, "used_by": used_by or None},
            cause=exc,
        ) from exc
