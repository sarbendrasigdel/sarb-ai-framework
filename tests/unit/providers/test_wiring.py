"""The path from a line of YAML to a working provider.

Every other test in this package exercises an adapter in isolation, with its client
handed to it. This one exercises the thing the framework actually promises: that
``provider: anthropic`` in a config file becomes an ``LLMProvider`` in an engine's
hands, and that swapping it for ``provider: openai`` is that one-line change and
nothing else.

It is also where the missing-extra experience is pinned down. That message is the
first thing a new contributor will see when they configure a provider they have not
installed, and it is the difference between running one command and reading a
traceback.
"""

from __future__ import annotations

import importlib

import pytest

from sarb.config import load_settings
from sarb.contracts import (
    EMBEDDERS,
    LLMS,
    EmbeddingKind,
    EmbeddingProvider,
    LLMProvider,
    LLMRequest,
    Message,
)
from sarb.di import build_container
from sarb.errors import ConfigurationError, MissingDependencyError, ProviderNotFoundError
from sarb.middleware import GuardedLLM
from sarb.providers.llm.fake import FakeProvider


def settings(**llms):
    return load_settings(overrides={"llms": llms})


def embedder_settings(**embedders):
    return load_settings(overrides={"embedders": embedders})


class TestResolution:
    async def test_a_line_of_yaml_becomes_a_provider(self):
        config = settings(
            default={"provider": "fake", "options": {"responses": ["Paris."]}},
        )

        async with build_container(config) as container:
            llm = await container.get(LLMS)

            # The engine's view: a port. Not a FakeProvider, not a vendor client.
            assert isinstance(llm, LLMProvider)
            response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))
            assert response.text == "Paris."

    async def test_named_components_are_the_point(self):
        # Production does not run one model. It runs a cheap one for routing, an
        # expensive one for synthesis, and a third as a judge.
        config = settings(
            default={"provider": "fake", "options": {"responses": ["from default"]}},
            judge={"provider": "fake", "options": {"responses": ["from judge"]}},
        )

        async with build_container(config) as container:
            default = await container.get(LLMS)
            judge = await container.get(LLMS, "judge")

            request = LLMRequest(messages=(Message.user("hi"),))
            assert (await default.generate(request)).text == "from default"
            assert (await judge.generate(request)).text == "from judge"

    async def test_a_provider_is_built_once(self):
        # It holds a connection pool. Building it per request would be a leak with a
        # deadline.
        config = settings(default={"provider": "fake", "options": {}})

        async with build_container(config) as container:
            first = await container.get(LLMS)
            second = await container.get(LLMS)

            assert first is second

    async def test_the_component_is_the_one_the_config_named(self):
        # What comes back is a *guarded* provider, not the bare adapter — the container
        # wraps every component in its retry and telemetry policy on the way out. So
        # the question "did I get the provider I asked for?" is answered by what the
        # guard delegates to, not by the class of the thing in your hand. An engine
        # cannot tell the difference, which is the entire point of the port.
        config = settings(default={"provider": "fake", "options": {"responses": ["from fake"]}})

        async with build_container(config) as container:
            llm = await container.get(LLMS)

            assert isinstance(llm, GuardedLLM)
            assert isinstance(llm._inner, FakeProvider)
            assert (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).text == (
                "from fake"
            )


class TestFailureIsUseful:
    async def test_a_missing_extra_names_the_command_to_run(self, monkeypatch):
        # The whole justification for a small core. If this message is not
        # actionable, the extras are just a way to break people's installs.
        #
        # The SDK is installed in this environment, so the absence has to be
        # simulated — but at the one place the framework actually reaches for it,
        # which means the message under test is the real one.
        real_import = importlib.import_module

        def missing(name: str, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("sarb.dependencies.importlib.import_module", missing)

        config = settings(
            default={"provider": "anthropic", "options": {"model": "claude-opus-4-8"}}
        )

        async with build_container(config) as container:
            with pytest.raises(MissingDependencyError) as caught:
                await container.get(LLMS)

        assert 'uv add "sarb[anthropic]"' in str(caught.value)
        assert caught.value.context["extra"] == "anthropic"
        # Not a ModuleNotFoundError from four frames down.
        assert "LLM provider 'anthropic'" in str(caught.value)

    async def test_a_typo_lists_what_was_available(self):
        # The cause is nearly always a typo, a missing extra, or a plugin that failed
        # to load — and all three are answered by the list.
        config = settings(default={"provider": "anthorpic", "options": {}})

        async with build_container(config) as container:
            with pytest.raises(ProviderNotFoundError) as caught:
                await container.get(LLMS)

        message = str(caught.value)
        assert "anthorpic" in message
        assert "anthropic" in message  # the one they meant

    async def test_a_bad_option_names_the_exact_yaml_key(self):
        # An operator should be able to find this in their config file without
        # reading a traceback.
        config = settings(default={"provider": "fake", "options": {"chunk_size": -1}})

        async with build_container(config) as container:
            with pytest.raises(ConfigurationError) as caught:
                await container.get(LLMS)

        assert "llms.default.options.chunk_size" in str(caught.value)

    async def test_an_unknown_option_is_rejected_rather_than_ignored(self):
        # `temprature: 0.7` must not leave the operator convinced they set something.
        config = settings(default={"provider": "fake", "options": {"temprature": 0.7}})

        async with build_container(config) as container:
            with pytest.raises(ConfigurationError, match="temprature"):
                await container.get(LLMS)


class TestSwapping:
    async def test_changing_the_vendor_is_a_config_change(self):
        # The claim the whole architecture rests on, asserted rather than assumed.
        # This function is what an engine looks like: it knows the port, and nothing
        # else. It is not recompiled, re-imported or re-tested when the config below
        # changes which provider it is handed.
        async def answer(llm: LLMProvider) -> str:
            return (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).text

        for responses in (["from A"], ["from B"]):
            config = settings(default={"provider": "fake", "options": {"responses": responses}})
            async with build_container(config) as container:
                assert await answer(await container.get(LLMS)) == responses[0]


class TestEmbedders:
    async def test_a_line_of_yaml_becomes_an_embedder(self):
        config = embedder_settings(
            default={"provider": "fake", "options": {"dimensions": 4}},
        )

        async with build_container(config) as container:
            embedder = await container.get(EMBEDDERS)

            # Again, the engine's view is the port. Not a FakeEmbeddingProvider.
            assert isinstance(embedder, EmbeddingProvider)
            assert embedder.dimensions == 4
            assert len(await embedder.embed(["hi"], kind=EmbeddingKind.QUERY)) == 1

    async def test_the_two_families_are_independent(self):
        # `fake` is a registry key in both `llms` and `embedders`, and they are
        # different providers. A registry that collapsed them would resolve one of
        # these to the wrong thing — and the port is structural, so it would not
        # necessarily fail loudly.
        config = load_settings(
            overrides={
                "llms": {"default": {"provider": "fake", "options": {"responses": ["Paris."]}}},
                "embedders": {"default": {"provider": "fake", "options": {"dimensions": 4}}},
            }
        )

        async with build_container(config) as container:
            llm = await container.get(LLMS)
            embedder = await container.get(EMBEDDERS)

            assert isinstance(llm, LLMProvider)
            assert isinstance(embedder, EmbeddingProvider)
            assert (await llm.generate(LLMRequest(messages=(Message.user("hi"),)))).text == "Paris."
            assert embedder.dimensions == 4

    async def test_a_bad_option_names_the_exact_yaml_key(self):
        config = embedder_settings(default={"provider": "fake", "options": {"dimensions": 0}})

        async with build_container(config) as container:
            with pytest.raises(ConfigurationError) as caught:
                await container.get(EMBEDDERS)

        assert "embedders.default.options.dimensions" in str(caught.value)


async def test_every_shipped_provider_is_registered():
    # A provider that exists but is not registered is a provider nobody can use, and
    # nothing else in the test suite would notice.
    config = settings(default={"provider": "fake", "options": {}})

    async with build_container(config) as container:
        llms = container._registry.providers("llms")
        embedders = container._registry.providers("embedders")

    assert set(llms) == {"anthropic", "openai", "gemini", "ollama", "fake"}
    assert set(embedders) == {"openai", "fastembed", "sentence-transformers", "fake"}
