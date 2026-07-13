"""One call to bring observability up, from the composition root."""

from __future__ import annotations

from sarb.config import Settings
from sarb.observability.logging import configure_logging, get_logger
from sarb.observability.telemetry import configure_telemetry

__all__ = ["configure_observability"]


def configure_observability(settings: Settings) -> None:
    """Configure logging and telemetry, then log the effective configuration.

    Call once, at startup, before anything else. Order matters: logging is
    installed first so that a failure to configure telemetry is itself reported
    through a working logger.

    The startup line is not decoration. Most production incidents that trace back
    to configuration are diagnosed in seconds when the effective config was logged
    and in hours when it was not — so the framework logs it rather than hoping
    each application remembers to. It is safe: every secret is masked by
    :meth:`~sarb.config.Settings.redacted`.

    Args:
        settings: The full application settings.

    Raises:
        MissingDependencyError: If telemetry is enabled without ``sarb[telemetry]``.

    Example:
        >>> settings = load_settings(config_dir="config", profile="production")
        >>> configure_observability(settings)
    """
    configure_logging(settings.logging)
    configure_telemetry(settings.telemetry)

    get_logger("sarb.startup").info(
        "application configured",
        app=settings.app.name,
        environment=settings.app.environment.value,
        telemetry=settings.telemetry.enabled,
        config=settings.redacted(),
    )
