"""Tests for the settings models: lookup, validation, redaction, immutability."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr

from sarb.config import (
    REDACTED,
    AppSettings,
    ComponentConfig,
    Environment,
    Settings,
    load_settings,
    parse_options,
)
from sarb.errors import ConfigurationError


class _OpenAIOptions(BaseModel):
    """Stand-in for an adapter's own options model."""

    model: str
    api_key: SecretStr
    temperature: float = 0.0


def test_defaults_load_without_any_configuration():
    settings = load_settings()

    assert settings.app.environment is Environment.DEVELOPMENT
    assert settings.server.port == 8000
    assert settings.llms == {}


def test_settings_are_frozen():
    settings = load_settings()

    with pytest.raises(Exception, match=r"frozen|immutable"):
        settings.app = AppSettings(name="mutated")  # type: ignore[misc]


class TestComponentLookup:
    @pytest.fixture
    def settings(self) -> Settings:
        return load_settings(
            overrides={
                "llms": {
                    "default": {"provider": "anthropic", "options": {"model": "claude-sonnet-5"}},
                    "fast": {"provider": "anthropic", "options": {"model": "claude-haiku-4-5"}},
                    "legacy": {"provider": "openai", "enabled": False},
                }
            }
        )

    def test_resolves_default_when_no_name_given(self, settings: Settings):
        assert settings.component("llms").provider == "anthropic"
        assert settings.component("llms").options["model"] == "claude-sonnet-5"

    def test_resolves_by_name(self, settings: Settings):
        assert settings.component("llms", "fast").options["model"] == "claude-haiku-4-5"

    def test_unknown_name_lists_what_is_available(self, settings: Settings):
        with pytest.raises(ConfigurationError) as exc_info:
            settings.component("llms", "judge")

        message = str(exc_info.value)
        assert "judge" in message
        # The whole point of the error: it answers its own question.
        assert "default" in message
        assert "fast" in message

    def test_unknown_family_lists_known_families(self, settings: Settings):
        with pytest.raises(ConfigurationError, match="Unknown component family"):
            settings.component("rerankers")

    def test_extras_provide_families_for_plugins(self):
        settings = load_settings(
            overrides={"extras": {"rerankers": {"default": {"provider": "cohere"}}}}
        )

        assert settings.component("rerankers").provider == "cohere"

    def test_enabled_components_filters_disabled(self, settings: Settings):
        assert set(settings.enabled_components("llms")) == {"default", "fast"}


class TestParseOptions:
    def test_produces_a_typed_model(self):
        config = ComponentConfig(
            provider="openai",
            options={"model": "gpt-4o", "api_key": "sk-test", "temperature": 0.7},
        )

        options = parse_options(config, _OpenAIOptions, component="llms.default")

        assert options.model == "gpt-4o"
        assert options.temperature == 0.7
        assert options.api_key.get_secret_value() == "sk-test"

    def test_error_names_the_config_path_not_the_python_field(self):
        config = ComponentConfig(provider="openai", options={"model": "gpt-4o"})

        with pytest.raises(ConfigurationError) as exc_info:
            parse_options(config, _OpenAIOptions, component="llms.default")

        # An operator must be able to find this in their YAML.
        assert "llms.default.options.api_key" in str(exc_info.value)

    def test_reports_every_bad_field_at_once(self):
        config = ComponentConfig(provider="openai", options={"temperature": "hot"})

        with pytest.raises(ConfigurationError) as exc_info:
            parse_options(config, _OpenAIOptions, component="llms.default")

        message = str(exc_info.value)
        assert "model" in message
        assert "api_key" in message
        assert "temperature" in message


class TestGuardrails:
    def test_debug_is_refused_in_production(self):
        with pytest.raises(ConfigurationError, match="debug"):
            load_settings(overrides={"app": {"debug": True, "environment": "production"}})

    def test_debug_is_fine_outside_production(self):
        settings = load_settings(overrides={"app": {"debug": True}})
        assert settings.app.debug

    def test_telemetry_without_an_endpoint_is_refused(self):
        with pytest.raises(ConfigurationError, match="endpoint"):
            load_settings(overrides={"telemetry": {"enabled": True}})

    def test_inverted_pool_bounds_are_refused(self):
        with pytest.raises(ConfigurationError, match="pool_min_size"):
            load_settings(overrides={"database": {"pool_min_size": 10, "pool_max_size": 2}})

    def test_unknown_top_level_key_is_refused(self):
        # A typo in config must not be silently ignored.
        with pytest.raises(ConfigurationError):
            load_settings(overrides={"loggging": {"level": "DEBUG"}})

    def test_validation_error_suggests_the_env_var_to_fix(self):
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(overrides={"server": {"port": 99999}})

        assert "SARB_SERVER__PORT" in str(exc_info.value)


class TestRedaction:
    def test_secret_str_fields_are_masked(self):
        settings = load_settings(
            overrides={"database": {"dsn": "postgresql://user:hunter2@db/app"}}
        )

        dumped = settings.redacted()

        assert dumped["database"]["dsn"] == REDACTED
        assert "hunter2" not in str(dumped)

    def test_secret_looking_option_keys_are_masked(self):
        settings = load_settings(
            overrides={
                "llms": {
                    "default": {
                        "provider": "anthropic",
                        "options": {"model": "claude-sonnet-5", "api_key": "sk-ant-live"},
                    }
                }
            }
        )

        dumped = settings.redacted()
        options = dumped["llms"]["default"]["options"]

        assert options["api_key"] == REDACTED
        assert options["model"] == "claude-sonnet-5"  # non-secrets survive
        assert "sk-ant-live" not in str(dumped)

    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "apiKey", "auth_token", "password", "client_secret", "dsn"],
    )
    def test_secret_key_detection_is_case_and_style_insensitive(self, key: str):
        config = ComponentConfig(provider="x", options={key: "leak-me"})

        assert config.redacted()["options"][key] == REDACTED
