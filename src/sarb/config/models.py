"""Typed settings models.

Design note — why component options are an untyped ``dict``
-----------------------------------------------------------
It is tempting to model provider configuration as a discriminated union::

    llm: OpenAIConfig | AnthropicConfig | OllamaConfig   # <-- do not do this

That union lives in the core, so every new provider — including one written by
a downstream team or a third-party plugin — requires editing a core file. The
kernel would accumulate knowledge of every vendor that ever existed, which is
precisely the coupling this framework is built to avoid.

Instead a component is ``{provider: str, options: dict}``. The core validates
only the envelope. Each adapter declares its own Pydantic options model and
validates its own slice at construction time via
:func:`sarb.config.parse_options`. The result: strong typing at the edge where
it matters, and a kernel that never learns a vendor's name.

Design note — why every family is a mapping
-------------------------------------------
Real systems run several models at once: a cheap one for routing, an expensive
one for synthesis, a third as an evaluation judge. So families are
``dict[str, ComponentConfig]`` keyed by a caller-chosen name, and the DI
container resolves by name, defaulting to ``"default"``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from sarb.config.sources import YamlSettingsSource
from sarb.contracts.component import DEFAULT_COMPONENT
from sarb.errors import ConfigurationError
from sarb.redaction import REDACTED, redact

__all__ = [
    "DEFAULT_COMPONENT",
    "REDACTED",
    "AppSettings",
    "ComponentConfig",
    "ComponentMap",
    "DatabaseSettings",
    "Environment",
    "LogLevel",
    "LoggingSettings",
    "PluginSettings",
    "RedisSettings",
    "ResilienceSettings",
    "ServerSettings",
    "Settings",
    "TelemetrySettings",
    "parse_options",
]


class Environment(StrEnum):
    """Deployment environment. Drives defaults that should differ by stage."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Standard log levels, mirroring :mod:`logging`."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class _Frozen(BaseModel):
    """Base for settings sections.

    Settings are frozen: a running process must not mutate its own
    configuration. Reconfiguration means building a new :class:`Settings` and
    swapping it in at a well-defined point, which keeps config changes
    observable instead of ambient.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ResilienceSettings(_Frozen):
    """How a component behaves when the thing behind it misbehaves.

    Every provider in this framework classifies its failures — a ``429`` becomes a
    :class:`~sarb.errors.RateLimitError` carrying ``retryable`` and ``retry_after``,
    a ``401`` becomes an :class:`~sarb.errors.AuthenticationError` that is retryable
    never. This is the policy that *reads* that classification.

    Attributes:
        max_attempts: Total attempts, including the first. ``1`` disables retry.
        initial_backoff: Seconds to wait after the first failure, before jitter.
        max_backoff: Ceiling on the computed wait. A provider's own ``Retry-After``
            is honoured even when it exceeds this — it is an instruction, not a
            suggestion.
        max_concurrency: How many calls may be in flight to this component at once.
            ``None`` means unlimited, which is the right answer for a local model
            that already limits itself and the wrong one for a hosted API.

    Why this lives in the kernel while ``options`` does not:
        The rule is that the kernel never learns a vendor's name. Resilience is
        entirely vendor-neutral — a rate limit means the same thing whoever imposed
        it — so the kernel is allowed to understand this, and a plugin nobody has
        written yet gets the same policy as Anthropic does, for free.

    Why retry defaults to on:
        Because the alternative is not "no retry". It is a retry loop hand-written
        into every engine by whoever hit a 429 first, and that loop will not honour
        ``Retry-After``, and it will hammer a provider that has just asked to be
        left alone.
    """

    max_attempts: int = Field(default=3, ge=1)
    initial_backoff: float = Field(default=0.5, gt=0)
    max_backoff: float = Field(default=30.0, gt=0)
    max_concurrency: int | None = Field(default=None, gt=0)


class ComponentConfig(_Frozen):
    """Declares one instance of a replaceable component.

    Attributes:
        provider: Registry key of the adapter to build, e.g. ``"anthropic"``.
            Resolved against the component registry at wiring time.
        enabled: When ``False`` the component is skipped during wiring. Lets an
            operator disable a provider without deleting its config block.
        options: Adapter-specific settings, validated by the adapter itself.
        resilience: Overrides the application-wide policy for this component only.
            Absent means inherit it.

    Example:
        ```yaml
        llms:
          default:
            provider: anthropic
            options:
              model: claude-sonnet-5
              api_key: ${ANTHROPIC_API_KEY}
          judge:
            provider: openai
            options:
              model: gpt-4o
            resilience:
              max_attempts: 1   # a judge that retries is a judge that bills triple
        ```
    """

    provider: str = Field(min_length=1, description="Registry key of the adapter to build.")
    enabled: bool = Field(default=True, description="Skip this component when false.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter-specific options, validated by the adapter, not the kernel.",
    )
    resilience: ResilienceSettings | None = Field(
        default=None,
        description="Per-component retry and concurrency policy. Inherits when absent.",
    )

    def redacted(self) -> dict[str, Any]:
        """Return a dict copy with secret-looking option values masked."""
        return {
            "provider": self.provider,
            "enabled": self.enabled,
            "options": redact(self.options),
        }


#: A named family of interchangeable components, e.g. every configured LLM.
ComponentMap = Annotated[
    dict[str, ComponentConfig],
    Field(default_factory=dict),
]


def parse_options[OptionsT: BaseModel](
    config: ComponentConfig,
    model: type[OptionsT],
    *,
    component: str = "<unknown>",
) -> OptionsT:
    """Validate a component's raw options against an adapter's options model.

    This is the seam that keeps vendor knowledge out of the kernel: the core
    carries ``options`` as an opaque dict, and the adapter turns it into a typed
    model at construction time.

    Args:
        config: The component config whose ``options`` should be parsed.
        model: The adapter's Pydantic options model.
        component: Dotted name used in error messages, e.g. ``"llms.default"``.

    Returns:
        A validated instance of ``model``.

    Raises:
        ConfigurationError: If the options do not satisfy the model. The message
            names the exact config path of every offending field, so an operator
            can find it in their YAML without reading a traceback.

    Example:
        >>> class OpenAIOptions(BaseModel):
        ...     model: str
        ...     api_key: SecretStr
        >>> opts = parse_options(config, OpenAIOptions, component="llms.default")
    """
    try:
        return model.model_validate(config.options)
    except ValidationError as exc:
        problems = "; ".join(
            f"{component}.options.{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid options for provider {config.provider!r}: {problems}",
            context={"component": component, "provider": config.provider},
            cause=exc,
        ) from exc


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


class AppSettings(_Frozen):
    """Identity of the running application."""

    name: str = Field(default="sarb-app", min_length=1)
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(
        default=False,
        description="Enables verbose errors and disables response caching. Never in production.",
    )

    @model_validator(mode="after")
    def _forbid_debug_in_production(self) -> Self:
        """Refuse to boot with debug enabled in production.

        Debug mode leaks internal error detail to API clients. This is the kind
        of misconfiguration that must fail at startup, not at exploit time.
        """
        if self.debug and self.environment is Environment.PRODUCTION:
            raise ValueError("app.debug must be false when app.environment is 'production'")
        return self


class ServerSettings(_Frozen):
    """HTTP surface. Consumed by the API module."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    root_path: str = Field(default="", description="Mount prefix when behind a reverse proxy.")
    cors_origins: tuple[str, ...] = ()
    request_timeout_s: float = Field(default=60.0, gt=0)


class LoggingSettings(_Frozen):
    """Log emission policy. Consumed by the observability module."""

    level: LogLevel = LogLevel.INFO
    format: Literal["json", "console"] = Field(
        default="json",
        description="'json' for machines, 'console' for humans. Default to json; a dev "
        "profile overrides it.",
    )
    redact_keys: tuple[str, ...] = Field(
        default=("api_key", "authorization", "password", "secret", "token"),
        description="Log field names whose values are masked before emission.",
    )


class TelemetrySettings(_Frozen):
    """OpenTelemetry export policy. Consumed by the observability module."""

    enabled: bool = False
    service_name: str = "sarb"
    endpoint: str | None = Field(default=None, description="OTLP/HTTP collector endpoint.")
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    capture_prompts: bool = Field(
        default=False,
        description="Attach prompt and completion text to spans. Off by default: prompts "
        "routinely contain user data and must not leak into a tracing backend by accident.",
    )

    @model_validator(mode="after")
    def _require_endpoint_when_enabled(self) -> Self:
        """Telemetry that is enabled but has nowhere to go is a silent failure."""
        if self.enabled and not self.endpoint:
            raise ValueError("telemetry.endpoint is required when telemetry.enabled is true")
        return self


class DatabaseSettings(_Frozen):
    """PostgreSQL connection policy."""

    dsn: SecretStr | None = Field(default=None, description="postgresql://user:pass@host/db")
    pool_min_size: int = Field(default=1, ge=0)
    pool_max_size: int = Field(default=10, ge=1)
    command_timeout_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _check_pool_bounds(self) -> Self:
        """A pool whose floor exceeds its ceiling deadlocks on first use."""
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("database.pool_min_size must be <= database.pool_max_size")
        return self


class RedisSettings(_Frozen):
    """Redis connection policy."""

    url: SecretStr | None = Field(default=None, description="redis://host:6379/0")
    max_connections: int = Field(default=20, ge=1)
    socket_timeout_s: float = Field(default=5.0, gt=0)


class ModelPrice(_Frozen):
    """What one model costs, per **million** tokens.

    Per million because that is the unit every vendor prints on its pricing page, and a
    config file you can fill in by copying the numbers you are looking at is a config
    file that gets filled in correctly. Per-token would mean transcribing ``0.000003``,
    and one misplaced zero is a cost dashboard that is wrong by a factor of ten with
    nothing to show for it.

    Attributes:
        input: Fresh input — tokens neither read from nor written to a cache.
        output: Generated tokens. Reasoning is billed at this rate too, and is already
            counted in ``output_tokens``.
        cached_input: Input served from a provider-side cache. Typically a tenth of
            ``input``.
        cache_write_input: Input written *into* the cache. Typically dearer than
            ``input`` — you pay a premium now to pay a tenth later.

    Why the two cache prices default to ``None`` rather than to a discount:
        An unstated cache price falls back to the full ``input`` price, so a config that
        does not mention caching **overstates** what caching costs rather than
        understating it. Cost figures that surprise you are the ones that surprise you
        pleasantly; the framework guessing "oh, that'll be a tenth" is how a bill arrives
        that nobody forecast.
    """

    input: float = Field(default=0.0, ge=0)
    output: float = Field(default=0.0, ge=0)
    cached_input: float | None = Field(default=None, ge=0)
    cache_write_input: float | None = Field(default=None, ge=0)


class PluginSettings(_Frozen):
    """Controls discovery of third-party components.

    Attributes:
        enabled: Load plugins advertised on the ``sarb.plugins`` entry point.
        allowlist: If non-empty, only these plugin names are loaded. Deployments
            that must not execute arbitrary installed code set this.
        denylist: Plugin names to skip. Applied after the allowlist.
    """

    enabled: bool = True
    allowlist: tuple[str, ...] = ()
    denylist: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


#: The component families the kernel knows about — one per port in
#: :mod:`sarb.contracts`. Declared once, and used to generate both the validator
#: and the "known families" error message, so the three cannot drift apart.
#:
#: A family a *plugin* introduces does not appear here; it lives in
#: :attr:`Settings.extras`, which is what keeps the kernel from having to grow a
#: field every time somebody adds a reranker.
_BUILTIN_FAMILIES = (
    "llms",
    "embedders",
    "vector_stores",
    "memories",
    "prompts",
    "loaders",
    "chunkers",
    "retrievers",
    "tools",
    "workflows",
    "storages",
)


def _secrets_dir_exists() -> bool:
    """Whether the conventional container secrets mount is present.

    Pointing pydantic-settings at a non-existent ``secrets_dir`` raises, so the
    directory is only wired in when a container actually mounted one.
    """
    return Path("/run/secrets").is_dir()


class Settings(BaseSettings):
    """Root configuration object for a ``sarb`` application.

    Values are layered, highest priority first:

    1. Explicit overrides passed to :func:`sarb.config.load_settings`
    2. Environment variables (``SARB_`` prefix, ``__`` nesting delimiter)
    3. A ``.env`` file
    4. YAML files: ``base.yaml`` then ``<profile>.yaml``, deep-merged
    5. Docker/Kubernetes secret files
    6. Field defaults

    Nested values follow the delimiter, so
    ``SARB_LLMS__DEFAULT__OPTIONS__MODEL=claude-sonnet-5`` sets exactly one leaf
    and leaves the rest of the YAML block intact.

    The object is frozen. To reconfigure, build a new one.
    """

    model_config = SettingsConfigDict(
        env_prefix="SARB_",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets" if _secrets_dir_exists() else None,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)

    #: The retry and concurrency policy every component inherits. A component may
    #: override it in its own block; see `ComponentConfig.resilience`.
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)

    #: Price list, as ``pricing[provider][model]``, in USD per million tokens.
    #:
    #: Configuration and not code, on purpose. A price list is the single most perishable
    #: fact in this whole system — vendors cut prices, add tiers, and negotiate contracts,
    #: and none of that should require a release of the framework to reflect. Nesting it
    #: under the provider also disambiguates the model name, which is not unique: an
    #: `llama3` served by Ollama on your own hardware is free, and the same name behind a
    #: hosted API is not.
    #:
    #: A model with no entry costs nothing that we know of, which is recorded as "no cost
    #: recorded" rather than as zero. See `sarb.middleware.cost`.
    pricing: dict[str, dict[str, ModelPrice]] = Field(default_factory=dict)

    # Component families: one per port in `sarb.contracts`, so every replaceable
    # abstraction has exactly one place it is configured. Each is a name ->
    # component mapping; see the module docstring for why these are maps rather
    # than single values.
    llms: ComponentMap
    embedders: ComponentMap
    vector_stores: ComponentMap
    memories: ComponentMap
    prompts: ComponentMap
    loaders: ComponentMap
    chunkers: ComponentMap
    retrievers: ComponentMap
    tools: ComponentMap
    workflows: ComponentMap
    storages: ComponentMap

    #: Escape hatch for component families introduced by plugins. A plugin that
    #: adds, say, a "rerankers" family reads its config from here without the
    #: kernel growing a field for it.
    extras: dict[str, dict[str, ComponentConfig]] = Field(default_factory=dict)

    @field_validator(*_BUILTIN_FAMILIES, mode="after")
    @classmethod
    def _reject_empty_names(cls, value: dict[str, ComponentConfig]) -> dict[str, ComponentConfig]:
        """Component names are used as registry and log keys; blanks corrupt both."""
        if any(not name.strip() for name in value):
            raise ValueError("component names must be non-empty")
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML layer between the ``.env`` file and secret files.

        Earlier entries win. Env beats YAML so that a container can override any
        single leaf without shipping a new config file — the property that makes
        the same image runnable in every environment.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    # -- lookup ------------------------------------------------------------- #

    def component(self, family: str, name: str = DEFAULT_COMPONENT) -> ComponentConfig:
        """Look up one component by family and name.

        Args:
            family: A built-in family (``"llms"``, ``"vector_stores"``, ...) or a
                plugin family registered under :attr:`extras`.
            name: The component name; defaults to ``"default"``.

        Returns:
            The matching component config.

        Raises:
            ConfigurationError: If the family or name is not configured. The
                message lists what *is* configured, which turns the single most
                common startup failure into a self-answering one.
        """
        members = self._family(family)
        try:
            return members[name]
        except KeyError:
            available = ", ".join(sorted(members)) or "<none configured>"
            raise ConfigurationError(
                f"No component named {name!r} in family {family!r}. Available: {available}",
                context={"family": family, "name": name, "available": sorted(members)},
            ) from None

    def _family(self, family: str) -> dict[str, ComponentConfig]:
        """Resolve a family to its mapping, checking built-ins then extras."""
        builtin = getattr(self, family, None)
        if isinstance(builtin, dict):
            return builtin
        if family in self.extras:
            return self.extras[family]
        known = sorted({*_BUILTIN_FAMILIES, *self.extras})
        raise ConfigurationError(
            f"Unknown component family {family!r}. Known families: {', '.join(known)}",
            context={"family": family, "known": known},
        )

    def enabled_components(self, family: str) -> dict[str, ComponentConfig]:
        """Return the enabled members of a family, in declaration order."""
        return {n: c for n, c in self._family(family).items() if c.enabled}

    # -- safe rendering ----------------------------------------------------- #

    def redacted(self) -> dict[str, Any]:
        """Dump settings with every secret masked.

        Safe to log at startup, and doing so is strongly recommended: most
        production incidents that trace back to config are diagnosed in seconds
        when the effective config was logged, and in hours when it was not.
        """
        raw = self.model_dump(mode="json")
        # redact() is structure-preserving, so a mapping in is a mapping out.
        return cast("dict[str, Any]", redact(raw))
