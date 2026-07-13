"""From a line of YAML to a working vector store.

Also the only test in the suite that exercises an **async factory**. ``pgvector``'s is
the first — a connection pool opens sockets, and sockets need an event loop — and the
container's support for it had never been used, which meant the guard was quietly
wrapping the coroutine instead of the component inside it. Nothing noticed, because
nothing had tried.
"""

from __future__ import annotations

import anyio
import anyio.lowlevel
import pytest

from sarb.config import load_settings
from sarb.contracts import VECTOR_STORES, Distance, VectorQuery, VectorRecord, VectorStore, eq
from sarb.di import build_container
from sarb.errors import ConfigurationError
from sarb.middleware import GuardedVectorStore
from sarb.plugins.registry import ComponentContext
from sarb.providers.vectorstore.memory import MemoryVectorStore, MemoryVectorStoreOptions


def settings(**stores):
    return load_settings(overrides={"vector_stores": stores})


class TestResolution:
    async def test_a_line_of_yaml_becomes_a_store(self):
        config = settings(default={"provider": "memory", "options": {}})

        async with build_container(config) as container:
            store = await container.get(VECTOR_STORES)

            # The engine's view: a port. Not a MemoryVectorStore, not a Qdrant client.
            assert isinstance(store, VectorStore)

            await store.ensure_collection("docs", dimensions=2, distance=Distance.COSINE)
            await store.upsert(
                "docs",
                [
                    VectorRecord(id="a", vector=[1.0, 0.0], metadata={"tenant": "acme"}),
                    VectorRecord(id="b", vector=[0.0, 1.0], metadata={"tenant": "globex"}),
                ],
            )

            hits = await store.search(
                "docs", VectorQuery(vector=[1.0, 0.0], top_k=5, where=eq("tenant", "acme"))
            )

            assert [hit.id for hit in hits] == ["a"]

    async def test_a_store_comes_back_guarded(self):
        # Vector stores get the same retry and telemetry as everything else, and for a
        # better reason than most: a dropped connection halfway through indexing leaves a
        # partial index, and a partial index is worse than none, because it looks like one.
        config = settings(default={"provider": "memory", "options": {}})

        async with build_container(config) as container:
            store = await container.get(VECTOR_STORES)

            assert isinstance(store, GuardedVectorStore)
            assert isinstance(store._inner, MemoryVectorStore)

    async def test_every_shipped_store_is_registered(self):
        config = settings(default={"provider": "memory", "options": {}})

        async with build_container(config) as container:
            available = container._registry.providers("vector_stores")

        assert set(available) == {"qdrant", "pgvector", "memory"}

    async def test_a_bad_option_names_the_exact_yaml_key(self):
        config = settings(default={"provider": "memory", "options": {"batch_size": 0}})

        async with build_container(config) as container:
            with pytest.raises(ConfigurationError) as caught:
                await container.get(VECTOR_STORES)

        assert "vector_stores.default.options.batch_size" in str(caught.value)


class TestAsyncFactories:
    async def test_the_container_awaits_a_factory_that_needs_an_event_loop(self):
        # `pgvector.build` is async, because `asyncpg.create_pool` returns a pool that is
        # not usable until it has been awaited. A synchronous factory could only hand
        # back something that fails on first use — moving the failure from startup, where
        # it belongs, to the first search a user runs.
        built: list[str] = []

        async def pooled(ctx: ComponentContext) -> MemoryVectorStore:
            """Stands in for `pgvector.build`: it has to await something."""
            await anyio.lowlevel.checkpoint()
            built.append(ctx.path)
            return MemoryVectorStore(MemoryVectorStoreOptions())

        config = settings(default={"provider": "pooled", "options": {}})

        async with build_container(config) as container:
            container._registry.add("vector_stores", "pooled", pooled)

            store = await container.get(VECTOR_STORES)

        assert built == ["vector_stores.default"]
        # And — the bug this test exists for — what came back is the *store*, wrapped.
        # Before the container awaited the factory, the guard was handed the coroutine
        # and faithfully wrapped that instead.
        assert isinstance(store, GuardedVectorStore)
        assert isinstance(store._inner, MemoryVectorStore)
