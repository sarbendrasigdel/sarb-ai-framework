"""Machinery every adapter needs, whatever family it belongs to.

An LLM API, an embedding API and a vector database are all HTTP services with the
same broad failure modes: they reject your key, they rate limit you, they time out,
they fall over. An engine that wants to back off on a ``429`` must not have to know
whose ``429`` it was — and it must not matter whether the ``429`` came from a chat
completion or an embedding batch.

So the status table lives here, once, and each family passes in the error class to
fall back to. Duplicating it per family would mean a bug in the error path — the one
path nobody exercises — appearing in some providers and not others.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sarb.errors import (
    AuthenticationError,
    InvalidInputError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)

__all__ = [
    "error_for_status",
    "retry_after_seconds",
    "unsupported",
]

#: The statuses that mean something specific to us. Everything else is just an
#: error; these four change what the caller is allowed to do next.
_UNAUTHORISED = (401, 403)
_RATE_LIMITED = 429
_TIMED_OUT = (408, 504)
_SERVER_ERROR = 500


def retry_after_seconds(headers: Mapping[str, Any] | None) -> float | None:
    """Read ``Retry-After`` out of a response, when the provider sent one.

    Returns:
        The delay in seconds, or ``None`` — including for the HTTP-date form of
        the header, which is legal and which we decline to guess at. A wrong
        number here is worse than no number: the resilience layer *prefers* this
        value over its own backoff curve, so a bad parse would hammer a provider
        that just asked to be left alone.
    """
    if not headers:
        return None

    raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if raw is None:
        return None

    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def error_for_status(
    status: int | None,
    message: str,
    *,
    provider: str,
    model: str,
    otherwise: type[ProviderError],
    retry_after: float | None = None,
    cause: BaseException | None = None,
) -> ProviderError:
    """Map an HTTP status onto the framework's error hierarchy.

    The classes returned here carry ``retryable``, which is the only thing the
    resilience layer reads. Everything else is context for the human.

    Args:
        status: The HTTP status, if there was a response at all.
        message: The provider's own message. Passed through rather than
            paraphrased — it is usually the most useful thing in the traceback.
        provider: The registry key, e.g. ``"anthropic"``.
        model: The model being called. Lands in ``context``, so failures are
            queryable by model.
        otherwise: The class to raise when the status says nothing more specific —
            :class:`~sarb.errors.LLMError` for a chat call,
            :class:`~sarb.errors.EmbeddingError` for an embedding one. The *kind*
            of call is the caller's knowledge; the *meaning* of a 429 is not.
        retry_after: What the provider asked for, if it asked.
        cause: The vendor exception being translated.
    """
    context: dict[str, Any] = {"model": model}
    if status is not None:
        context["status_code"] = status

    if status in _UNAUTHORISED:
        return AuthenticationError(message, provider=provider, context=context, cause=cause)

    if status == _RATE_LIMITED:
        return RateLimitError(
            message, retry_after=retry_after, provider=provider, context=context, cause=cause
        )

    if status in _TIMED_OUT:
        return ProviderTimeoutError(message, provider=provider, context=context, cause=cause)

    if status is not None and status >= _SERVER_ERROR:
        # 500, 502, 503, and Anthropic's 529 "overloaded". Transient by definition.
        return otherwise(message, provider=provider, context=context, cause=cause, retryable=True)

    return otherwise(message, provider=provider, context=context, cause=cause)


def unsupported(provider: str, feature: str, *, hint: str = "") -> InvalidInputError:
    """The request asks for something this provider cannot express.

    Raised, never ignored. Quietly dropping a ``response_format`` leaves the caller
    believing output is schema-constrained when it is not, and they will find out
    downstream, in a JSON parse error, a long way from the cause. Silently doing less
    than you were asked is the failure mode that costs the most to diagnose.
    """
    message = f"provider {provider!r} cannot express {feature}"
    if hint:
        message = f"{message}. {hint}"
    return InvalidInputError(message, context={"provider": provider, "feature": feature})
