"""Tests for layering: file profiles, env interpolation, and source precedence.

Precedence is the property most likely to be broken by a well-meaning refactor
and least likely to be noticed until production reads the wrong value, so it is
asserted directly rather than inferred.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sarb.config import (
    Settings,
    current_settings,
    deep_merge,
    expand_env,
    load_settings,
    use_settings,
)
from sarb.errors import ConfigurationError


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A config directory with a base file and two profiles."""
    (tmp_path / "base.yaml").write_text(
        """
        app:
          name: base-app
        server:
          port: 8000
          host: 127.0.0.1
        llms:
          default:
            provider: anthropic
            options:
              model: claude-sonnet-5
              temperature: 0.0
        """,
        encoding="utf-8",
    )
    (tmp_path / "production.yaml").write_text(
        """
        app:
          name: prod-app
          environment: production
        server:
          host: 0.0.0.0
        llms:
          default:
            options:
              temperature: 0.2
        """,
        encoding="utf-8",
    )
    (tmp_path / "interpolated.yaml").write_text(
        """
        llms:
          default:
            provider: anthropic
            options:
              api_key: ${TEST_ANTHROPIC_KEY}
              region: ${TEST_REGION:-us-east-1}
        """,
        encoding="utf-8",
    )
    return tmp_path


class TestProfiles:
    def test_base_only(self, config_dir: Path):
        settings = load_settings(config_dir=config_dir)

        assert settings.app.name == "base-app"
        assert settings.server.port == 8000

    def test_profile_deep_merges_onto_base(self, config_dir: Path):
        settings = load_settings(config_dir=config_dir, profile="production")

        assert settings.app.name == "prod-app"  # overridden
        assert settings.server.host == "0.0.0.0"  # overridden
        assert settings.server.port == 8000  # inherited from base

        options = settings.component("llms").options
        assert options["temperature"] == 0.2  # overridden by profile
        assert options["model"] == "claude-sonnet-5"  # inherited, not clobbered

    def test_missing_profile_is_an_error_not_a_silent_fallback(self, config_dir: Path):
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(config_dir=config_dir, profile="staging")

        message = str(exc_info.value)
        assert "staging" in message
        assert "production" in message  # tells you what does exist

    def test_missing_config_dir_is_an_error(self, tmp_path: Path):
        with pytest.raises(ConfigurationError, match="does not exist"):
            load_settings(config_dir=tmp_path / "nope")

    def test_malformed_yaml_names_the_file(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text("app: [unclosed", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Malformed YAML"):
            load_settings(config_dir=tmp_path)


class TestInterpolation:
    def test_expands_from_the_environment(self, config_dir: Path):
        settings = load_settings(
            config_dir=config_dir,
            profile="interpolated",
            environ={"TEST_ANTHROPIC_KEY": "sk-ant-xyz"},
        )

        options = settings.component("llms").options
        assert options["api_key"] == "sk-ant-xyz"
        assert options["region"] == "us-east-1"  # default branch

    def test_explicit_value_beats_the_default_branch(self, config_dir: Path):
        settings = load_settings(
            config_dir=config_dir,
            profile="interpolated",
            environ={"TEST_ANTHROPIC_KEY": "k", "TEST_REGION": "eu-west-1"},
        )

        assert settings.component("llms").options["region"] == "eu-west-1"

    def test_missing_variable_fails_at_load_not_at_first_use(self, config_dir: Path):
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(config_dir=config_dir, profile="interpolated", environ={})

        assert "TEST_ANTHROPIC_KEY" in str(exc_info.value)

    def test_double_dollar_escapes_a_literal_placeholder(self):
        # Prompt templates legitimately contain ${...}; they must survive.
        assert expand_env("$${NOT_A_VAR}", {}) == "${NOT_A_VAR}"

    def test_expansion_recurses_through_lists_and_dicts(self):
        result = expand_env({"a": ["${X}", {"b": "${Y:-d}"}]}, {"X": "1"})

        assert result == {"a": ["1", {"b": "d"}]}


class TestPrecedence:
    """Overrides > env > yaml > defaults, asserted one boundary at a time."""

    def test_env_beats_yaml(self, config_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SARB_SERVER__PORT", "9999")

        settings = load_settings(config_dir=config_dir)

        assert settings.server.port == 9999

    def test_env_overrides_one_leaf_without_clobbering_its_siblings(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # This is the property that lets one container image run everywhere.
        monkeypatch.setenv("SARB_LLMS__DEFAULT__OPTIONS__TEMPERATURE", "0.9")

        settings = load_settings(config_dir=config_dir)
        options = settings.component("llms").options

        assert options["temperature"] == "0.9"
        assert options["model"] == "claude-sonnet-5"  # sibling survived

    def test_overrides_beat_env(self, config_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SARB_SERVER__PORT", "9999")

        settings = load_settings(config_dir=config_dir, overrides={"server": {"port": 7777}})

        assert settings.server.port == 7777

    def test_yaml_beats_defaults(self, config_dir: Path):
        assert load_settings(config_dir=config_dir).app.name == "base-app"

    def test_defaults_apply_when_nothing_else_does(self):
        assert load_settings().app.name == "sarb-app"

    def test_dotenv_is_read_when_present(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("SARB_APP__NAME=from-dotenv\n", encoding="utf-8")

        assert load_settings(env_file=env_file).app.name == "from-dotenv"

    def test_missing_dotenv_is_harmless(self, tmp_path: Path):
        assert load_settings(env_file=tmp_path / "absent.env").app.name == "sarb-app"


class TestDeepMerge:
    def test_merges_nested_mappings_key_by_key(self):
        assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}

    def test_lists_replace_rather_than_concatenate(self):
        assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_inputs_are_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"y": 2}})

        assert base == {"a": {"x": 1}}


class TestAmbientSettings:
    def test_reading_before_installing_is_an_error(self):
        with pytest.raises(ConfigurationError, match="No settings are installed"):
            current_settings()

    def test_use_settings_installs_and_restores(self):
        settings = load_settings(overrides={"app": {"name": "scoped"}})

        with use_settings(settings):
            assert current_settings().app.name == "scoped"

        with pytest.raises(ConfigurationError):
            current_settings()

    def test_nesting_restores_the_outer_settings(self):
        outer = load_settings(overrides={"app": {"name": "outer"}})
        inner = load_settings(overrides={"app": {"name": "inner"}})

        with use_settings(outer):
            with use_settings(inner):
                assert current_settings().app.name == "inner"
            assert current_settings().app.name == "outer"


def test_loading_is_pure(config_dir: Path):
    """Two loads with the same inputs produce equal settings and no shared state."""
    first = load_settings(config_dir=config_dir, profile="production")
    second = load_settings(config_dir=config_dir, profile="production")

    assert first == second
    assert first is not second
    assert isinstance(first, Settings)
