"""Masking secrets before they reach a log line, a span, or a config dump.

One implementation, used by both :mod:`sarb.config` and
:mod:`sarb.observability`. They need identical behaviour — an ``api_key`` must be
masked whether it is being printed at startup or attached to a log record — and
two copies of a redaction rule is exactly the kind of duplication that ends with
one of them quietly falling out of date.

How a key is matched
--------------------
The key is normalised — ``apiKey``, ``API_KEY`` and ``x-api-key`` all become
``api_key`` — and then matched **exactly, or on a trailing ``_hint`` segment**.
So the hint ``key`` masks ``api_key``, ``openai_api_key`` and a bare ``key``.

It is deliberately *not* a substring match, which is the obvious implementation
and is wrong in this domain. ``"token"`` as a substring also matches ``tokens``,
``input_tokens`` and ``token_count`` — so a naive substring rule silently redacts
every token count in every log line, destroying the single most important metric
an LLM framework emits. Suffix matching keeps ``access_token`` masked and
``input_tokens`` visible.

The bias is still toward over-masking where the two conflict: a needlessly masked
field costs a developer a moment of confusion, while a missed one puts a live
credential in a log aggregator that a dozen people can read and that retains it
for a year.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import SecretStr

__all__ = [
    "DEFAULT_SECRET_HINTS",
    "REDACTED",
    "is_secret_key",
    "redact",
]

#: Placeholder substituted for a secret value.
REDACTED = "***"

#: Key segments that mark a value as a credential. Matched exactly, or as a
#: trailing ``_<hint>`` segment of the normalised key.
DEFAULT_SECRET_HINTS = frozenset(
    {
        "key",
        "secret",
        "token",
        "password",
        "passwd",
        "credential",
        "credentials",
        "authorization",
        "auth",
        "dsn",
        "apikey",
    }
)

#: Separators and camelCase boundaries, both rewritten to ``_``.
_SEPARATORS = re.compile(r"[-. ]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalise(key: str) -> str:
    """Reduce any casing or separator style to lower snake_case.

    ``apiKey``, ``API-KEY`` and ``api.key`` all become ``api_key``, so one hint
    covers every style a vendor SDK or a YAML file might use.
    """
    key = _SEPARATORS.sub("_", key)
    key = _CAMEL_BOUNDARY.sub("_", key)
    return key.lower()


#: How deep to walk a nested structure before giving up. Log fields are supposed
#: to be shallow; anything deeper is far more likely to be a mistake (or a cycle)
#: than a genuine value worth rendering.
_MAX_DEPTH = 8


def is_secret_key(key: str, hints: frozenset[str] = DEFAULT_SECRET_HINTS) -> bool:
    """Whether a key's *name* suggests its value is a credential.

    The key is normalised, then matched exactly or on a trailing ``_<hint>``
    segment. See the module docstring for why this is not a substring match.

    Args:
        key: The field or option name, in any casing or style.
        hints: Key segments that mark a key as secret.

    Returns:
        True if the key names a credential.

    Example:
        >>> is_secret_key("ANTHROPIC_API_KEY")
        True
        >>> is_secret_key("input_tokens")     # not a credential
        False
    """
    normalised = _normalise(key)
    if normalised in hints:
        return True
    return any(normalised.endswith(f"_{hint}") for hint in hints)


def redact(
    value: Any,
    *,
    hints: frozenset[str] = DEFAULT_SECRET_HINTS,
    _depth: int = 0,
) -> Any:
    """Recursively mask secrets in an arbitrary structure.

    Two things are masked: any value whose *key* looks like a credential, and any
    :class:`~pydantic.SecretStr` anywhere, regardless of the key it sits under.

    Args:
        value: Any JSON-like structure — dict, list, tuple, or scalar.
        hints: Substrings that mark a key as secret.

    Returns:
        A masked copy. The input is never modified.

    Example:
        >>> redact({"model": "claude-sonnet-5", "api_key": "sk-live-abc"})
        {'model': 'claude-sonnet-5', 'api_key': '***'}
    """
    if isinstance(value, SecretStr):
        return REDACTED

    if _depth >= _MAX_DEPTH:
        # Too deep to be a real log field, and possibly a cycle. Do not recurse
        # further, and do not emit the raw value: it has not been inspected.
        return REDACTED

    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if is_secret_key(str(key), hints)
                else redact(item, hints=hints, _depth=_depth + 1)
            )
            for key, item in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [redact(item, hints=hints, _depth=_depth + 1) for item in value]

    return value
