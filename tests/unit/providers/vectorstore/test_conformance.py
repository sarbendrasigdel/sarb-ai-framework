"""One suite, every backend, the same answers.

This is the most valuable file in the module, and the reason is worth stating plainly.

Each adapter compiles the filter algebra to something different — Qdrant's DSL, a SQL
predicate, a Python expression — and a unit test of a *compiler* can only check the
compiler against the same understanding that produced it. If I misread what ``ne``
should do when the field is absent, I will write the compiler wrong and then write a
test that agrees with it, and both will be wrong together, for years.

So the semantics are pinned **once**, behaviourally, against the in-process store, and
every other backend is required to give the same answers to the same questions. That is
what caught Qdrant quietly matching records that had no such field at all.

The stores that need a server are marked ``integration`` and skipped by default. The
suite is written so that turning them on is a fixture change and nothing else.
"""

from __future__ import annotations

import pytest

from sarb.contracts import (
    Distance,
    SearchHit,
    VectorQuery,
    VectorRecord,
    eq,
    exists,
    gt,
    gte,
    in_,
    lt,
    lte,
    ne,
    not_in,
)
from sarb.errors import InvalidInputError, VectorStoreError
from sarb.providers.vectorstore.memory import MemoryVectorStore, MemoryVectorStoreOptions

COLLECTION = "docs"

#: A corpus with deliberate holes in it. `orphan` has no `tenant` and no `year`, which
#: is what every interesting filter question turns out to be about.
CORPUS = [
    VectorRecord(id="a", vector=[1.0, 0.0], metadata={"tenant": "acme", "year": 2024}),
    VectorRecord(id="b", vector=[0.9, 0.1], metadata={"tenant": "acme", "year": 2020}),
    VectorRecord(id="c", vector=[0.0, 1.0], metadata={"tenant": "globex", "year": 2024}),
    VectorRecord(id="orphan", vector=[0.7, 0.7], metadata={"title": "no tenant, no year"}),
]


@pytest.fixture
async def store():
    """The reference store. Every other backend must agree with it."""
    store = MemoryVectorStore(MemoryVectorStoreOptions())
    await store.ensure_collection(COLLECTION, dimensions=2, distance=Distance.COSINE)
    await store.upsert(COLLECTION, CORPUS)
    return store


async def ids_matching(store, where) -> set[str]:
    """Which records a filter selects.

    Asked through `count` *and* `search`, because an adapter that compiles a filter
    twice can compile it differently twice.
    """
    hits = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=100, where=where))
    by_search = {hit.id for hit in hits}

    assert await store.count(COLLECTION, where=where) == len(by_search), (
        "count and search disagree about the same filter"
    )
    return by_search


class TestTheAlgebra:
    async def test_no_filter_matches_everything(self, store):
        assert await ids_matching(store, None) == {"a", "b", "c", "orphan"}

    async def test_eq(self, store):
        assert await ids_matching(store, eq("tenant", "acme")) == {"a", "b"}

    async def test_eq_does_not_match_a_record_that_lacks_the_field(self, store):
        # The one that matters most. `eq` is how tenant isolation is enforced, and a
        # record with no tenant must never satisfy it — otherwise every document that
        # forgot to say who it belonged to is visible to everyone.
        assert "orphan" not in await ids_matching(store, eq("tenant", "acme"))

    async def test_ne_requires_the_field_to_be_present(self, store):
        # `ne` is a *comparison*, and a comparison against a field that is not there is
        # false. So `ne("tenant", "acme")` means "has a tenant, and it is not acme" —
        # not "is not known to be acme".
        #
        # This is the exact assertion that caught Qdrant, whose natural spelling of `ne`
        # (a bare `must_not`) also matched `orphan`.
        assert await ids_matching(store, ne("tenant", "acme")) == {"c"}

    async def test_not_is_a_real_negation_and_therefore_is_not_ne(self, store):
        # And this is the flip side, which is the subtlety worth reading twice.
        # `Not(eq(...))` negates the whole comparison, so it *does* match the record
        # with no tenant: that record is not one whose tenant is acme.
        #
        # `ne` and `Not(eq)` are therefore different questions, deliberately. One asks
        # "does it differ?", the other asks "is it not the case that it matches?" — and
        # the difference is exactly the records that are missing the field.
        assert await ids_matching(store, ~eq("tenant", "acme")) == {"c", "orphan"}

    async def test_in_and_not_in(self, store):
        assert await ids_matching(store, in_("tenant", ["acme", "globex"])) == {"a", "b", "c"}
        assert await ids_matching(store, not_in("tenant", ["acme"])) == {"c"}

    async def test_ordering(self, store):
        assert await ids_matching(store, gte("year", 2024)) == {"a", "c"}
        assert await ids_matching(store, gt("year", 2020)) == {"a", "c"}
        assert await ids_matching(store, lt("year", 2024)) == {"b"}
        assert await ids_matching(store, lte("year", 2020)) == {"b"}

    async def test_ordering_does_not_match_a_record_that_lacks_the_field(self, store):
        assert "orphan" not in await ids_matching(store, gte("year", 1900))

    async def test_exists(self, store):
        assert await ids_matching(store, exists("tenant")) == {"a", "b", "c"}
        assert await ids_matching(store, exists("tenant", present=False)) == {"orphan"}

    async def test_and(self, store):
        assert await ids_matching(store, eq("tenant", "acme") & gte("year", 2024)) == {"a"}

    async def test_or(self, store):
        assert await ids_matching(store, eq("tenant", "globex") | eq("year", 2020)) == {"b", "c"}

    async def test_nested(self, store):
        where = eq("tenant", "acme") & ~eq("year", 2020)

        assert await ids_matching(store, where) == {"a"}


class TestSearch:
    async def test_hits_come_back_best_first(self, store):
        hits = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=4))

        assert [hit.id for hit in hits] == ["a", "b", "orphan", "c"]

    async def test_scores_descend(self, store):
        # The port's promise: higher is better, whatever the metric. Retrieval code sorts
        # one way, always, without asking which distance the store was built with.
        hits = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=4))
        scores = [hit.score for hit in hits]

        assert scores == sorted(scores, reverse=True)

    async def test_top_k_cuts(self, store):
        hits = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=2))

        assert [hit.id for hit in hits] == ["a", "b"]

    async def test_the_filter_is_applied_before_the_cut_not_after(self, store):
        # The single most common bug in hand-rolled vector search: fetch the top k, then
        # filter them, and quietly return fewer than k. It does not look like a bug. It
        # looks like "retrieval is a bit weak", which is why it survives for years.
        #
        # `c` is the *worst* match for this query and the only globex record. A
        # post-filter over the top 1 would return nothing at all.
        hits = await store.search(
            COLLECTION,
            VectorQuery(vector=[1.0, 0.0], top_k=1, where=eq("tenant", "globex")),
        )

        assert [hit.id for hit in hits] == ["c"]

    async def test_min_score_drops_the_weak(self, store):
        hits = await store.search(
            COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=10, min_score=0.9)
        )

        assert [hit.id for hit in hits] == ["a", "b"]

    async def test_vectors_are_withheld_unless_asked_for(self, store):
        # They are large, and almost nobody wants them. Returning them by default means
        # every search moves megabytes nobody reads.
        [hit] = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=1))
        assert hit.vector is None

        [hit] = await store.search(
            COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=1, include_vectors=True)
        )
        assert hit.vector == [1.0, 0.0]

    async def test_metadata_comes_back_with_the_hit(self, store):
        [hit] = await store.search(COLLECTION, VectorQuery(vector=[1.0, 0.0], top_k=1))

        assert hit.metadata["tenant"] == "acme"
        assert isinstance(hit, SearchHit)


class TestWriting:
    async def test_upsert_replaces_rather_than_duplicates(self, store):
        # The property that makes re-indexing a changed document safe. Without it, every
        # re-index doubles the corpus and retrieval slowly fills with stale copies.
        await store.upsert(
            COLLECTION,
            [VectorRecord(id="a", vector=[0.0, 1.0], metadata={"tenant": "acme", "year": 2025})],
        )

        assert await store.count(COLLECTION) == 4
        [record] = await store.fetch(COLLECTION, ["a"])
        assert record.metadata["year"] == 2025

    async def test_fetch_omits_what_is_not_there(self, store):
        records = await store.fetch(COLLECTION, ["a", "nope"])

        assert [r.id for r in records] == ["a"]

    async def test_delete_by_id(self, store):
        assert await store.delete(COLLECTION, ids=["a", "nope"]) == 1
        assert await store.count(COLLECTION) == 3

    async def test_delete_by_filter(self, store):
        assert await store.delete(COLLECTION, where=eq("tenant", "acme")) == 2
        assert await ids_matching(store, None) == {"c", "orphan"}

    async def test_delete_needs_a_selector(self, store):
        # `delete(collection)` with no argument is one slip away from erasing a
        # production index, so there is no such call to slip into.
        with pytest.raises(InvalidInputError):
            await store.delete(COLLECTION)

    async def test_delete_refuses_both_selectors(self, store):
        # Ambiguous — union or intersection? — and two adapters would guess differently.
        with pytest.raises(InvalidInputError):
            await store.delete(COLLECTION, ids=["a"], where=eq("tenant", "acme"))


class TestCollections:
    async def test_ensure_is_idempotent(self, store):
        await store.ensure_collection(COLLECTION, dimensions=2, distance=Distance.COSINE)

        assert await store.count(COLLECTION) == 4  # nothing was wiped

    async def test_it_refuses_to_reshape_an_existing_collection(self, store):
        # Silently writing 768-dimension vectors into a 1536-dimension collection, or
        # changing the metric under a live index, corrupts retrieval in a way that shows
        # up as "the answers got worse" weeks later and is never traced back.
        with pytest.raises(VectorStoreError, match="cannot be reshaped"):
            await store.ensure_collection(COLLECTION, dimensions=768)

    async def test_a_changed_metric_is_also_a_reshape(self, store):
        with pytest.raises(VectorStoreError, match="cannot be reshaped"):
            await store.ensure_collection(COLLECTION, dimensions=2, distance=Distance.DOT)

    async def test_a_mis_sized_vector_is_refused(self, store):
        # What a model change that was not carried through to the index looks like.
        with pytest.raises(VectorStoreError, match="3-dimensional vector"):
            await store.upsert(COLLECTION, [VectorRecord(id="x", vector=[1.0, 2.0, 3.0])])

    async def test_dropping_is_idempotent(self, store):
        await store.drop_collection(COLLECTION)
        await store.drop_collection(COLLECTION)

        with pytest.raises(VectorStoreError, match="no collection"):
            await store.count(COLLECTION)
