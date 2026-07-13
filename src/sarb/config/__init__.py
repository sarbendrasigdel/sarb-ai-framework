"""Configuration: layered, typed, validated at startup.

Public API::

    from sarb.config import Settings, load_settings, parse_options

    settings = load_settings(config_dir="config", profile="production")
    llm_config = settings.component("llms", "default")

See :mod:`sarb.config.models` for why component options are deliberately
untyped in the kernel, and :mod:`sarb.config.sources` for the file layer.
"""

from sarb.config.loader import current_settings, load_settings, use_settings
from sarb.config.models import (
    DEFAULT_COMPONENT,
    REDACTED,
    AppSettings,
    ComponentConfig,
    ComponentMap,
    DatabaseSettings,
    Environment,
    LoggingSettings,
    LogLevel,
    ModelPrice,
    PluginSettings,
    RedisSettings,
    ResilienceSettings,
    ServerSettings,
    Settings,
    TelemetrySettings,
    parse_options,
)
from sarb.config.sources import FileContext, deep_merge, expand_env

__all__ = [
    "DEFAULT_COMPONENT",
    "REDACTED",
    "AppSettings",
    "ComponentConfig",
    "ComponentMap",
    "DatabaseSettings",
    "Environment",
    "FileContext",
    "LogLevel",
    "LoggingSettings",
    "ModelPrice",
    "PluginSettings",
    "RedisSettings",
    "ResilienceSettings",
    "ServerSettings",
    "Settings",
    "TelemetrySettings",
    "current_settings",
    "deep_merge",
    "expand_env",
    "load_settings",
    "parse_options",
    "use_settings",
]
