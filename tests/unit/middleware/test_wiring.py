"""From a line of YAML to a component that survives a rate limit.

Every other test in this package builds a guard by hand. This one asserts the thing the
framework actually promises: that an operator who writes ``max_attempts: 5`` gets five
attempts, in an application whose code contains no retry loop, against a provider that
has never heard of one.
"""

from __future__ import annotations

import pytest

from sarb.config import ResilienceSettings, load_settings
from sarb.contracts import EMBEDDERS, LLMS, LLMProvider, LLMRequest, Message
from sarb.di import build_container
from sarb.errors import RateLimitError
from sarb.middleware import GuardedEmbedder, GuardedLLM, policy_for

ASK = LLMRequest(messages=(Message.user("What is the capital of France?"),))


def settings(**overrides):
    return load_settings(overrides=overrides)


class TestTheContainerAppliesIt:
    async def test_a_component_comes_back_guarded(self):
        config = settings(llms={"default": {"provider": "fake", "options": {}}})

        async with build_container(config) as container:
            assert isinstance(await container.get(LLMS), GuardedLLM)

    async def test_every_family_with_a_wrapper_gets_one(self):
        config = settings(
            llms={"default": {"provider": "fake", "options": {}}},
            embedders={"default": {"provider": "fake", "options": {}}},
        )

        async with build_container(config) as container:
            assert isinstance(await container.get(LLMS), GuardedLLM)
            assert isinstance(await container.get(EMBEDDERS), GuardedEmbedder)

    async def test_the_guard_is_invisible_to_the_caller(self):
        # An engine sees a port. It cannot tell whether anything is wrapped around the
        # provider, and it must not be able to — the day it can, the guard has become
        # something an engine has to know about, and the whole arrangement is pointless.
        config = settings(llms={"default": {"provider": "fake", "options": {}}})

        async with build_container(config) as container:
            llm = await container.get(LLMS)

            assert isinstance(llm, LLMProvider)
            assert llm.model == "fake"
            assert llm.capabilities.streaming


class TestRetryFromConfig:
    async def test_a_rate_limited_call_recovers_without_the_caller_knowing(self):
        # The whole module, in one assertion. The application code below is what an
        # engine looks like — a single `generate`, no loop, no try, no backoff — and it
        # survives two rate limits because a config file said it should.
        config = settings(
            resilience={"max_attempts": 3, "initial_backoff": 0.001},
            llms={
                "default": {
                    "provider": "fake",
                    "options": {
                        "responses": ["Paris."],
                        "fail_with": "rate_limit",
                        "fail_times": 2,
                    },
                }
            },
        )

        async with build_container(config) as container:
            llm = await container.get(LLMS)

            assert (await llm.generate(ASK)).text == "Paris."

    async def test_it_still_gives_up_eventually(self):
        # A retry policy that never surrenders is a hang, not a policy.
        config = settings(
            resilience={"max_attempts": 2, "initial_backoff": 0.001},
            llms={
                "default": {
                    "provider": "fake",
                    "options": {"fail_with": "rate_limit"},  # fails forever
                }
            },
        )

        async with build_container(config) as container:
            with pytest.raises(RateLimitError):
                await (await container.get(LLMS)).generate(ASK)

    async def test_embedding_recovers_too(self):
        # Where retry matters most. Indexing fires hundreds of requests in a burst, so
        # indexing is what gets throttled — and a rate limit that reaches the caller
        # here leaves a half-built index, which is worse than no index, because it looks
        # like one.
        config = settings(
            resilience={"max_attempts": 3, "initial_backoff": 0.001},
            embedders={
                "default": {
                    "provider": "fake",
                    "options": {"dimensions": 4, "fail_with": "rate_limit", "fail_times": 2},
                }
            },
        )

        async with build_container(config) as container:
            vectors = (await (await container.get(EMBEDDERS)).embed(["a", "b"])).vectors

        assert len(vectors) == 2


class TestPolicyResolution:
    def test_a_component_inherits_the_application_default(self):
        config = settings(
            resilience={"max_attempts": 7},
            llms={"default": {"provider": "fake", "options": {}}},
        )

        assert policy_for(config.llms["default"], config).max_attempts == 7

    def test_a_component_may_override_it(self):
        # A judge that retries is a judge that bills triple.
        config = settings(
            resilience={"max_attempts": 7},
            llms={
                "default": {"provider": "fake", "options": {}},
                "judge": {"provider": "fake", "options": {}, "resilience": {"max_attempts": 1}},
            },
        )

        assert policy_for(config.llms["default"], config).max_attempts == 7
        assert policy_for(config.llms["judge"], config).max_attempts == 1

    def test_an_override_replaces_rather_than_merges(self):
        # Deliberate. A partial merge reads as "these settings, plus whatever the
        # default happens to be" — a sentence nobody can act on when the default later
        # changes underneath them.
        config = settings(
            resilience={"max_attempts": 7, "initial_backoff": 9.0},
            llms={"judge": {"provider": "fake", "options": {}, "resilience": {"max_attempts": 1}}},
        )

        policy = policy_for(config.llms["judge"], config)

        assert policy.max_attempts == 1
        assert policy.initial_backoff == ResilienceSettings().initial_backoff

    def test_retry_is_on_by_default(self):
        # Because the alternative is not "no retry". It is a retry loop hand-written
        # into every engine by whoever hit a 429 first, and that loop will not honour
        # Retry-After.
        assert load_settings().resilience.max_attempts > 1
