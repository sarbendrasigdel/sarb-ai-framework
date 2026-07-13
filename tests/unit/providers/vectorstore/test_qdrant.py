"""Tests for the Qdrant adapter.

The client is injected, so nothing here needs a server. The filters are asserted as
Qdrant's **own** model objects rather than as dicts — a hand-written dict can be wrong in
exactly the way the compiler is wrong, and the two would then agree with each other
forever.

Read ``TestFilterCompilation`` first. It is the part of this adapter that can be
catastrophically wrong while looking completely fine.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import models

# `qdrant_client.models` re-exports two different classes called `QueryResponse` — the
# local-inference one wins the name, and the one `query_points` actually returns is
# shadowed. Importing the real one by path rather than discovering this from a
# ValidationError at some later date.
from qdrant_client.http.models.models import QueryResponse

from sarb.config import ComponentConfig
from sarb.contracts import (
    Distance,
    VectorQuery,
    VectorRecord,
    VectorStore,
    eq,
    exists,
    gte,
    in_,
    lt,
    ne,
    not_in,
)
from sarb.errors import InvalidInputError, MissingDependencyError, VectorStoreError
from sarb.providers.vectorstore.qdrant import (
    ID_KEY,
    QdrantOptions,
    QdrantStore,
    build,
    point_id,
)


class FakeClient:
    """Records what would have gone to Qdrant, and returns what we tell it to."""

    def __init__(self, **results):
        self.results = results
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def __getattr__(self, name: str):
        """Every method Qdrant's client has, and any it grows: recorded, not called."""

        async def call(*args, **kwargs):
            self.calls.append((name, kwargs))
            result = self.results.get(name)
            if isinstance(result, BaseException):
                raise result
            return result

        return call

    async def close(self):
        self.closed = True


def store(**results) -> QdrantStore:
    return QdrantStore(FakeClient(**results), QdrantOptions())  # type: ignore[arg-type]


def sent(s, method: str):
    """The kwargs of the last call to ``method``."""
    return next(kwargs for name, kwargs in reversed(s._client.calls) if name == method)


def calls(s):
    """Every call the adapter would have made, in order."""
    return s._client.calls


def closed(s):
    """Whether the adapter released its client."""
    return s._client.closed


class TestFilterCompilation:
    def test_no_filter_is_no_filter(self):
        assert store().compile(None) is None

    def test_eq(self):
        compiled = store().compile(eq("tenant", "acme"))

        assert compiled == models.Filter(
            must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme"))]
        )

    def test_ne_excludes_records_that_lack_the_field(self):
        # The bug the conformance suite caught. A bare `must_not` on the value also
        # matches every record with no such field — a record with no tenant is, after
        # all, not equal to "acme". The algebra says a comparison against an absent
        # field is false, so the empty-condition is what makes Qdrant agree with the
        # reference implementation.
        compiled = store().compile(ne("tenant", "acme"))

        [inner] = compiled.must
        assert inner.must_not == [
            models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="tenant")),
        ]

    def test_not_in_excludes_records_that_lack_the_field(self):
        compiled = store().compile(not_in("tenant", ["acme"]))

        [inner] = compiled.must
        assert models.IsEmptyCondition(is_empty=models.PayloadField(key="tenant")) in inner.must_not

    def test_in(self):
        compiled = store().compile(in_("tenant", ["acme", "globex"]))

        assert compiled.must == [
            models.FieldCondition(key="tenant", match=models.MatchAny(any=["acme", "globex"]))
        ]

    def test_exists_is_the_negation_of_is_empty(self):
        # Qdrant has no "has key". It has "is empty", so presence is its negation.
        present = store().compile(exists("tenant"))
        absent = store().compile(exists("tenant", present=False))

        empty = models.IsEmptyCondition(is_empty=models.PayloadField(key="tenant"))
        assert present.must[0].must_not == [empty]
        assert absent.must[0].must == [empty]

    def test_ordering_becomes_a_range(self):
        compiled = store().compile(gte("year", 2020))

        assert compiled.must == [models.FieldCondition(key="year", range=models.Range(gte=2020.0))]

    def test_ordering_a_string_raises_rather_than_matching_nothing(self):
        # The one place the algebra is wider than the backend. Qdrant's Range holds
        # numbers only, so `lt("name", "m")` is meaningless to it — and sending it
        # anyway would match nothing at all, which looks exactly like "there are no such
        # documents" and is therefore never investigated.
        with pytest.raises(InvalidInputError, match="cannot express"):
            store().compile(lt("name", "m"))

    def test_and_or_not_nest(self):
        compiled = store().compile(eq("tenant", "acme") & ~eq("year", 2020))

        [conjunction] = compiled.must
        assert len(conjunction.must) == 2
        assert conjunction.must[1].must_not  # the negation

    def test_or_becomes_should(self):
        compiled = store().compile(eq("a", 1) | eq("b", 2))

        [disjunction] = compiled.must
        assert len(disjunction.should) == 2


class TestIds:
    def test_a_caller_id_becomes_a_uuid(self):
        # Qdrant point ids must be an unsigned integer or a UUID. "contract-42#chunk-3"
        # is exactly what a chunker produces, passes the client's type check, and is then
        # rejected by the server.
        minted = point_id("contract-42#chunk-3")

        assert len(minted) == 36
        assert minted.count("-") == 4

    def test_the_same_id_always_mints_the_same_uuid(self):
        # The property that makes re-indexing a changed document *replace* it rather than
        # add a second copy. A random id here would silently double the corpus on every
        # re-index.
        assert point_id("chunk-1") == point_id("chunk-1")
        assert point_id("chunk-1") != point_id("chunk-2")

    async def test_the_real_id_is_carried_in_the_payload(self):
        s = store(upsert=None)

        await s.upsert("docs", [VectorRecord(id="doc#1", vector=[1.0], metadata={"x": 1})])

        [point] = sent(s, "upsert")["points"]
        assert point.id == point_id("doc#1")
        assert point.payload[ID_KEY] == "doc#1"
        assert point.payload["x"] == 1

    async def test_the_real_id_comes_back_out(self):
        hit = models.ScoredPoint(
            id=point_id("doc#1"),
            version=1,
            score=0.9,
            payload={ID_KEY: "doc#1", "tenant": "acme"},
        )
        s = store(query_points=QueryResponse(points=[hit]))

        [found] = await s.search("docs", VectorQuery(vector=[1.0]))

        assert found.id == "doc#1"
        # And the bookkeeping key does not leak into the caller's metadata.
        assert found.metadata == {"tenant": "acme"}


class TestSearch:
    async def test_the_filter_goes_to_the_index_not_to_a_post_filter(self):
        # Filtering the top-k after the fact silently returns fewer than top_k, which
        # reads as "retrieval is a bit weak" rather than as a bug.
        s = store(query_points=QueryResponse(points=[]))

        await s.search("docs", VectorQuery(vector=[1.0], top_k=5, where=eq("tenant", "acme")))

        params = sent(s, "query_points")
        assert params["limit"] == 5
        assert params["query_filter"] is not None

    async def test_euclidean_scores_are_inverted_so_that_higher_is_better(self):
        # The port promises higher-is-better whatever the metric. Handing back a raw
        # Euclidean distance would rank the *furthest* documents first, and raise
        # nothing: the search would work, and the answers would be wrong.
        s = store(
            collection_exists=False,
            create_collection=None,
            query_points=QueryResponse(
                points=[
                    models.ScoredPoint(id=point_id("near"), version=1, score=0.0, payload={}),
                    models.ScoredPoint(id=point_id("far"), version=1, score=3.0, payload={}),
                ]
            ),
        )
        await s.ensure_collection("docs", dimensions=1, distance=Distance.EUCLIDEAN)

        hits = await s.search("docs", VectorQuery(vector=[1.0]))

        assert hits[0].score > hits[1].score  # distance 0 beats distance 3

    async def test_cosine_scores_pass_through_untouched(self):
        s = store(
            collection_exists=False,
            create_collection=None,
            query_points=QueryResponse(
                points=[models.ScoredPoint(id=point_id("a"), version=1, score=0.87, payload={})]
            ),
        )
        await s.ensure_collection("docs", dimensions=1, distance=Distance.COSINE)

        [hit] = await s.search("docs", VectorQuery(vector=[1.0]))

        assert hit.score == 0.87


class TestCollections:
    async def test_a_missing_collection_is_created(self):
        s = store(collection_exists=False, create_collection=None)

        await s.ensure_collection("docs", dimensions=384, distance=Distance.COSINE)

        params = sent(s, "create_collection")
        assert params["vectors_config"] == models.VectorParams(size=384, distance="Cosine")

    async def test_an_existing_collection_of_the_right_shape_is_left_alone(self):
        s = store(collection_exists=True, get_collection=_collection_info(384, "Cosine"))

        await s.ensure_collection("docs", dimensions=384, distance=Distance.COSINE)

        assert not any(name == "create_collection" for name, _ in calls(s))

    async def test_a_reshape_is_refused(self):
        s = store(collection_exists=True, get_collection=_collection_info(768, "Cosine"))

        with pytest.raises(VectorStoreError, match="cannot be reshaped"):
            await s.ensure_collection("docs", dimensions=384)


class TestWriting:
    async def test_a_large_write_is_split_into_batches(self):
        s = QdrantStore(
            FakeClient(upsert=None),  # type: ignore[arg-type]
            QdrantOptions(batch_size=2),
        )

        await s.upsert("docs", [VectorRecord(id=f"{i}", vector=[1.0]) for i in range(5)])

        upserts = [kwargs for name, kwargs in calls(s) if name == "upsert"]
        assert [len(u["points"]) for u in upserts] == [2, 2, 1]

    async def test_an_empty_write_touches_nothing(self):
        s = store()

        await s.upsert("docs", [])

        assert calls(s) == []

    async def test_delete_needs_exactly_one_selector(self):
        s = store()

        with pytest.raises(InvalidInputError):
            await s.delete("docs")


class TestErrors:
    async def test_a_backend_failure_becomes_a_vector_store_error(self):
        s = store(query_points=RuntimeError("connection reset"))

        with pytest.raises(VectorStoreError, match="connection reset") as caught:
            await s.search("docs", VectorQuery(vector=[1.0]))

        # A vector database is a network service. Resets, leader elections and restarts
        # are transient, and the middleware knows what to do with a retryable error.
        assert caught.value.retryable

    async def test_the_collection_is_queryable_on_the_error(self):
        s = store(query_points=RuntimeError("boom"))

        with pytest.raises(VectorStoreError) as caught:
            await s.search("contracts", VectorQuery(vector=[1.0]))

        assert caught.value.context["collection"] == "contracts"


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(store(), VectorStore)

    async def test_closing_releases_the_client(self):
        s = store()

        await s.aclose()

        assert closed(s)

    def test_the_factory_needs_the_extra(self, monkeypatch):
        real = importlib.import_module

        def missing(name, *args, **kwargs):
            if name == "qdrant_client":
                raise ImportError("no qdrant_client")
            return real(name, *args, **kwargs)

        monkeypatch.setattr("sarb.dependencies.importlib.import_module", missing)

        class Ctx:
            config = ComponentConfig(provider="qdrant")
            path = "vector_stores.default"

        with pytest.raises(MissingDependencyError, match='uv add "sarb\\[qdrant\\]"'):
            build(Ctx())  # type: ignore[arg-type]


def _collection_info(size: int, distance: str):
    """What ``get_collection`` returns, shaped the way the adapter reads it."""
    return models.CollectionInfo(
        status=models.CollectionStatus.GREEN,
        optimizer_status=models.OptimizersStatusOneOf.OK,
        segments_count=1,
        payload_schema={},
        config=models.CollectionConfig(
            params=models.CollectionParams(
                vectors=models.VectorParams(size=size, distance=distance)
            ),
            hnsw_config=models.HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000),
            optimizer_config=models.OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                flush_interval_sec=5,
                max_optimization_threads=1,
            ),
            wal_config=models.WalConfig(wal_capacity_mb=32, wal_segments_ahead=0),
        ),
    )
