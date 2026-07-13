"""Plugin system: which adapters exist, and where they came from.

The registry maps a config key to a factory::

    llms:
      default:
        provider: anthropic     # <- registry key

    registry.lookup("llms", "anthropic").factory

Third parties extend the framework through the ``sarb.plugins`` entry-point
group, receiving the registry to populate. See
:mod:`sarb.plugins.discovery` for the contract and the security policy around it.

This module holds no instances. Building and owning components is the job of
:mod:`sarb.di`.
"""

from sarb.plugins.discovery import ENTRY_POINT_GROUP, build_registry, load_plugins
from sarb.plugins.registry import (
    ComponentContext,
    ComponentFactory,
    ComponentRegistry,
    PluginRegistrar,
    Registration,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "ComponentContext",
    "ComponentFactory",
    "ComponentRegistry",
    "PluginRegistrar",
    "Registration",
    "build_registry",
    "load_plugins",
]
