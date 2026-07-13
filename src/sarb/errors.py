"""The framework's exception hierarchy.

Every error raised by ``sarb`` derives from :class:`SarbError`. This gives
application code a single, reliable seam for handling framework failures::

    try:
        answer = await agent.run(question)
    except SarbError as exc:
        logger.error("agent failed", extra=exc.context)

Two properties make this hierarchy useful in production rather than decorative:

``context``
    A structured dict carried alongside the message. Log formatters and the
    telemetry layer emit it as fields, so failures are queryable ("show me every
    ``RateLimitError`` for ``provider=openai``") instead of grep-able.

``retryable``
    Whether retrying the identical call could plausibly succeed. Adapters set
    this; the resilience layer reads it. Business logic never has to know that a
    vendor's 429 means "wait" while its 400 means "give up".

Adapters are responsible for translating vendor exceptions into these types.
That translation is the whole point: business logic must be able to handle a
failed embedding call without importing an exception class from a vendor SDK.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentError",
    "AuthenticationError",
    "ChunkingError",
    "ConfigurationError",
    "DocumentError",
    "EmbeddingError",
    "InvalidInputError",
    "LLMError",
    "MemoryProviderError",
    "MissingDependencyError",
    "PluginError",
    "PromptError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderTimeoutError",
    "RateLimitError",
    "RegistryError",
    "RetrievalError",
    "SarbError",
    "StorageError",
    "ToolError",
    "VectorStoreError",
    "WorkflowError",
]


class SarbError(Exception):
    """Base class for every error raised by the framework.

    Args:
        message: Human-readable description of what went wrong.
        context: Structured key/value pairs describing the failure. Emitted as
            log fields and span attributes; must not contain secrets.
        cause: The underlying exception, when this error wraps one. Prefer this
            over bare ``raise ... from`` when the cause is also worth reporting.
        retryable: Whether an identical retry could plausibly succeed. Defaults
            to ``False``: failures are permanent unless a subclass or an adapter
            explicitly says otherwise.
    """

    #: Default retry semantics for the class; instances may override.
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})
        self.retryable = self.default_retryable if retryable is None else retryable
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Render the message followed by sorted context pairs, if any."""
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={self.context[k]!r}" for k in sorted(self.context))
        return f"{self.message} ({details})"

    def __repr__(self) -> str:
        """Render an unambiguous form for test failures and logs."""
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


# --------------------------------------------------------------------------- #
# Kernel: configuration, wiring, plugins
# --------------------------------------------------------------------------- #


class ConfigurationError(SarbError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised at startup wherever possible. A process that boots with bad config
    should fail immediately and loudly, not on the first request at 3am.
    """


class RegistryError(SarbError):
    """A component registry was used incorrectly (e.g. a duplicate key)."""


class ProviderNotFoundError(RegistryError):
    """No provider is registered under the requested family and key.

    Usually means a missing extra (``pip install sarb[qdrant]``), a typo in
    config, or a plugin that failed to load.
    """


class PluginError(SarbError):
    """A plugin failed to load, validate, or register its components."""


class MissingDependencyError(SarbError):
    """A component needs an optional extra that is not installed.

    Raised by :func:`sarb.dependencies.import_optional`, with the exact install
    command in the message. Distinct from :class:`ProviderNotFoundError`, which
    means the provider key is unknown rather than uninstalled — an operator
    facing a typo and an operator facing a missing wheel need different advice.
    """


class InvalidInputError(SarbError):
    """Caller-supplied arguments are invalid.

    Distinct from :class:`pydantic.ValidationError`, which is a schema-level
    failure. This is the framework rejecting a semantically invalid request.
    """


# --------------------------------------------------------------------------- #
# Adapters: anything crossing a process boundary
# --------------------------------------------------------------------------- #


class ProviderError(SarbError):
    """Base class for failures originating in an external provider.

    Adapters translate vendor exceptions into this hierarchy so that engines can
    react to *categories* of failure without importing vendor SDKs.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
    ) -> None:
        merged = dict(context or {})
        if provider is not None:
            merged.setdefault("provider", provider)
        super().__init__(message, context=merged, cause=cause, retryable=retryable)
        self.provider = provider


class AuthenticationError(ProviderError):
    """Credentials were rejected by a provider. Never retryable."""


class RateLimitError(ProviderError):
    """A provider applied rate limiting or quota enforcement.

    Args:
        retry_after: Seconds the provider asked the caller to wait, when it said
            so. The resilience layer prefers this over its own backoff curve.
    """

    default_retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged = dict(context or {})
        if retry_after is not None:
            merged.setdefault("retry_after", retry_after)
        super().__init__(message, provider=provider, context=merged, cause=cause)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """A provider did not respond within the configured deadline."""

    default_retryable = True


class LLMError(ProviderError):
    """A language-model call failed."""


class EmbeddingError(ProviderError):
    """An embedding call failed."""


class VectorStoreError(ProviderError):
    """A vector-store operation failed."""


class StorageError(ProviderError):
    """A blob, relational, or key-value storage operation failed."""


class MemoryProviderError(ProviderError):
    """A memory-provider operation failed.

    Named for the port rather than the concept, because ``MemoryError`` is a
    builtin that means something entirely different and must stay reachable.
    """


# --------------------------------------------------------------------------- #
# Engines: the framework's own use-cases
# --------------------------------------------------------------------------- #


class PromptError(SarbError):
    """A prompt could not be resolved, rendered, or validated."""


class DocumentError(SarbError):
    """A document could not be loaded or parsed."""


class ChunkingError(SarbError):
    """A document could not be split into chunks."""


class RetrievalError(SarbError):
    """A retrieval strategy failed to produce results."""


class ToolError(SarbError):
    """A tool failed during execution.

    Tool failures are frequently *recoverable by the model*: the agent loop may
    feed the error back as a tool result and let the model try again. See
    ``sarb.engines.tools`` for that policy.
    """


class AgentError(SarbError):
    """An agent run failed or violated its limits (e.g. max iterations)."""


class WorkflowError(SarbError):
    """A workflow failed to execute or is structurally invalid (e.g. a cycle)."""
