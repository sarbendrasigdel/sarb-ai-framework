"""Shared test fixtures.

The one non-obvious thing here is environment isolation. ``Settings`` reads
``SARB_*`` environment variables by design, which means a developer who has
exported ``SARB_APP__NAME`` in their shell would see different test results than
CI — the worst class of flake, because it only reproduces on one machine. Every
test therefore runs against a scrubbed environment unless it opts back in.
"""

from __future__ import annotations

import os

import pytest

_FRAMEWORK_ENV_PREFIX = "SARB_"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``SARB_*`` variable so the ambient shell cannot alter tests.

    No teardown: ``monkeypatch`` restores the environment itself.
    """
    for key in [k for k in os.environ if k.startswith(_FRAMEWORK_ENV_PREFIX)]:
        monkeypatch.delenv(key, raising=False)
