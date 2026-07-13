"""Executable enforcement of the architecture.

Clean Architecture survives exactly as long as someone is checking. Code review
is not that someone: the import that breaks the model is one line in a large,
otherwise-good pull request, and it always looks reasonable in isolation.

So the rules are tests. They parse every module under ``src/sarb`` and assert
two invariants:

1. **The dependency rule.** Inner rings must not import outer rings. Most
   importantly: ``engines`` must never import ``providers``. The day a retrieval
   engine imports ``sarb.providers.vectorstores.qdrant`` "just for a type hint"
   is the day Qdrant becomes impossible to replace.

2. **Vendor containment.** A vendor SDK may only be imported inside the adapter
   that adapts it. This is what makes ``pip install sarb`` work without every
   vendor SDK present, and what guarantees a vendor swap touches one directory.

Both are checked statically with :mod:`ast` — no imports are executed, so the
test runs without any optional extra installed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SRC / "sarb"

#: Ring -> the ``sarb.*`` subpackages it is forbidden from importing.
#: Read each entry as: "code in <key> must not know that <values> exist."
FORBIDDEN_IMPORTS: dict[str, frozenset[str]] = {
    # The innermost ring. Ports and models know nothing about anything.
    "contracts": frozenset({"engines", "providers", "api", "cli", "di", "plugins", "config"}),
    # Use-cases depend on ports, never on the adapters behind them.
    "engines": frozenset({"providers", "api", "cli"}),
    # Adapters implement ports; they must not reach sideways into use-cases.
    "providers": frozenset({"engines", "api", "cli"}),
    # Middleware decorates a port. It must work on an adapter it has never seen —
    # including one from a plugin — so the moment it can name a provider, it is
    # capable of special-casing one, and a guard that special-cases Anthropic is a
    # guard that silently does nothing for the provider you write next year.
    "middleware": frozenset({"providers", "engines", "api", "cli"}),
    # The kernel is depended upon; it does not depend back.
    "config": frozenset({"engines", "providers", "api", "cli"}),
    "observability": frozenset({"engines", "providers", "api", "cli"}),
    "di": frozenset({"engines", "api", "cli"}),
    "plugins": frozenset({"engines", "api", "cli"}),
}

#: Third-party package -> the only ``sarb`` subpackage allowed to import it.
#: A vendor SDK outside its adapter is a leak: it means some non-adapter module
#: has learned a vendor's data model, and the vendor is no longer replaceable.
VENDOR_CONTAINMENT: dict[str, str] = {
    "openai": "sarb.providers",
    "anthropic": "sarb.providers",
    "google": "sarb.providers",
    "ollama": "sarb.providers",
    "qdrant_client": "sarb.providers",
    "fastembed": "sarb.providers",
    "sentence_transformers": "sarb.providers",
    "asyncpg": "sarb.providers",
    "pgvector": "sarb.providers",
    "redis": "sarb.providers",
    "httpx": "sarb.providers",
    "fastapi": "sarb.api",
    "starlette": "sarb.api",
    "uvicorn": "sarb.api",
    "typer": "sarb.cli",
    "rich": "sarb.cli",
    "opentelemetry": "sarb.observability",
}


def _modules() -> list[Path]:
    """Every Python module in the package."""
    return sorted(PACKAGE.rglob("*.py"))


def _module_name(path: Path) -> str:
    """Dotted module name for a file, e.g. ``sarb.config.models``."""
    relative = path.relative_to(SRC).with_suffix("")
    parts = [p for p in relative.parts if p != "__init__"]
    return ".".join(parts)


def _imported_roots(path: Path) -> Iterator[tuple[str, int]]:
    """Yield the top-level package of every import in a module, with its line.

    Relative imports are resolved against the module's own package, so
    ``from ..providers import x`` inside ``sarb.engines.agent`` is caught just as
    ``import sarb.providers`` would be.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = _module_name(path).split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import x` inside a package: drop `level` trailing parts.
                base = package_parts[: len(package_parts) - node.level + 1]
                resolved = [*base, *(node.module.split(".") if node.module else [])]
                yield ".".join(resolved), node.lineno
            elif node.module:
                yield node.module, node.lineno


def _subpackage(path: Path) -> str | None:
    """The ``sarb`` subpackage a module belongs to, e.g. ``"engines"``."""
    relative = path.relative_to(PACKAGE)
    return relative.parts[0] if len(relative.parts) > 1 else None


@pytest.mark.parametrize("module_path", _modules(), ids=_module_name)
def test_module_respects_the_dependency_rule(module_path: Path) -> None:
    """No module imports a ring it is forbidden to know about."""
    ring = _subpackage(module_path)
    forbidden = FORBIDDEN_IMPORTS.get(ring or "", frozenset())
    if not forbidden:
        return

    violations = [
        f"{_module_name(module_path)}:{line} imports {imported!r}"
        for imported, line in _imported_roots(module_path)
        if imported.startswith("sarb.") and imported.split(".")[1] in forbidden
    ]

    assert not violations, (
        f"Dependency rule violated. Code in 'sarb.{ring}' must not import "
        f"{sorted(forbidden)}:\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize("module_path", _modules(), ids=_module_name)
def test_vendor_sdks_stay_inside_their_adapters(module_path: Path) -> None:
    """No vendor SDK is imported outside the subpackage that adapts it."""
    module = _module_name(module_path)

    violations = []
    for imported, line in _imported_roots(module_path):
        root = imported.split(".")[0]
        allowed_prefix = VENDOR_CONTAINMENT.get(root)
        if allowed_prefix and not module.startswith(allowed_prefix):
            violations.append(
                f"{module}:{line} imports {imported!r} (allowed only in {allowed_prefix}.*)"
            )

    assert not violations, (
        "Vendor SDK leaked outside its adapter. Business logic must depend on a port, "
        "not a vendor:\n  " + "\n  ".join(violations)
    )


def test_the_rules_actually_cover_the_package() -> None:
    """Guard against the rules silently applying to nothing.

    If the package is refactored and a ring is renamed, the parametrized tests
    above would keep passing while checking nothing at all. This asserts the
    rules still match reality.
    """
    existing = {p.name for p in PACKAGE.iterdir() if p.is_dir() and not p.name.startswith("_")}
    described = set(FORBIDDEN_IMPORTS)

    unknown = existing - described
    assert not unknown, (
        f"Subpackage(s) {sorted(unknown)} have no architecture rule. Add them to "
        "FORBIDDEN_IMPORTS (an empty frozenset is a valid, deliberate answer)."
    )
