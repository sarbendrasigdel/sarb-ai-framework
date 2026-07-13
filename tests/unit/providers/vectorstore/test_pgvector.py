"""Tests for the pgvector adapter.

The SQL *is* the adapter, so the SQL is what gets asserted. The pool is injected and
records every statement it would have run, which means these tests check the text that
would genuinely have gone to Postgres rather than a paraphrase of it.

Two sections are worth reading before changing anything here: ``TestInjection``, because
a table name is the one string in this adapter that cannot be a bind parameter, and
``TestThreeValuedLogic``, because SQL's ``NULL`` does not mean ``False`` and the filter
algebra says it must.
"""

from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")

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
)
from sarb.errors import InvalidInputError, VectorStoreError
from sarb.providers.vectorstore.pgvector import (
    PgVectorOptions,
    PgVectorStore,
    table_for,
)

DSN = "postgresql://localhost/test"


class FakeConnection:
    def __init__(self, owner):
        self._owner = owner

    async def execute(self, sql, *params):
        self._owner.statements.append((sql, params))
        return self._owner.status

    async def fetch(self, sql, *params):
        self._owner.statements.append((sql, params))
        return self._owner.rows

    async def fetchrow(self, sql, *params):
        self._owner.statements.append((sql, params))
        return self._owner.row

    async def executemany(self, sql, args):
        self._owner.statements.append((sql, args))

    def transaction(self):
        return _Nothing()


class _Nothing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakePool:
    """Records every statement, and hands back whatever the test scripted."""

    def __init__(self, rows=None, row=None, status="DELETE 0"):
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.rows = rows or []
        self.row = row
        self.status = status
        self.closed = False

    def acquire(self):
        return _Acquire(FakeConnection(self))

    async def close(self):
        self.closed = True


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


def store(pool: FakePool | None = None, **options) -> PgVectorStore:
    return PgVectorStore(pool or FakePool(), PgVectorOptions(dsn=DSN, **options))


def sql_of(s) -> list[str]:
    """Every statement, whitespace-normalised for legibility."""
    return [" ".join(text.split()) for text, _ in s._pool.statements]


class TestInjection:
    def test_a_reasonable_name_is_quoted(self):
        assert table_for("documents") == '"sarb_documents"'

    @pytest.mark.parametrize(
        "hostile",
        [
            'docs"; DROP TABLE users; --',
            "docs; DELETE FROM sarb_collections",
            "Documents",  # capitals: not in the allowed set
            "",
            "1docs",  # must start with a letter
            "docs-2024",  # a hyphen is not an underscore
            "a" * 60,
        ],
    )
    def test_a_hostile_or_odd_name_is_refused_rather_than_escaped(self, hostile):
        # This is the one string in the adapter that reaches the query text, because
        # Postgres cannot bind an identifier the way it binds a value. So it is *proven*
        # safe rather than escaped and hoped for: a name that is not obviously fine is
        # not used at all.
        with pytest.raises(InvalidInputError, match="not usable as a Postgres table"):
            table_for(hostile)

    async def test_a_metadata_key_is_a_bind_parameter_not_a_string(self):
        # Field names go through `metadata -> $1`, which Postgres *will* parameterise
        # even though it will not parameterise an identifier. So a metadata key called
        # `'; DROP TABLE users; --` is just a key that matches nothing.
        s = store()
        clause = s.compile(eq("'; DROP TABLE users; --", "x"))

        assert "DROP" not in clause.text
        assert "'; DROP TABLE users; --" in clause.params


class TestThreeValuedLogic:
    async def test_every_comparison_carries_a_presence_guard(self):
        # `metadata->>'tenant' <> 'acme'` on a row with no tenant is NULL, not FALSE.
        # And `NOT NULL` is NULL — so a negated filter would silently fail to match the
        # rows that are missing the field, which is not what the algebra says.
        # `metadata ? 'tenant' AND ...` forces a real FALSE, and NOT FALSE is TRUE.
        clause = store().compile(ne("tenant", "acme"))

        assert "metadata ? $1" in clause.text
        assert "AND" in clause.text

    async def test_not_negates_something_that_is_never_null(self):
        clause = store().compile(~eq("tenant", "acme"))

        # The guard is *inside* the NOT, which is what makes the negation total.
        assert clause.text.startswith("(NOT (metadata ? $1")

    async def test_exists_is_the_key_operator(self):
        assert store().compile(exists("tenant")).text == "metadata ? $1"
        assert store().compile(exists("tenant", present=False)).text == "(NOT metadata ? $1)"


class TestFilterCompilation:
    def test_eq_compares_jsonb_so_that_types_survive(self):
        # `metadata->>'year'` is *text*. Comparing it to 2024 as text would make "9"
        # greater than "10". Comparing the jsonb values keeps the number a number.
        clause = store().compile(eq("year", 2024))

        assert "metadata -> $1 = $2::jsonb" in clause.text
        assert clause.params == ["year", "2024"]

    def test_a_numeric_ordering_casts(self):
        clause = store().compile(gte("year", 2020))

        assert "(metadata ->> $1)::numeric >= $2" in clause.text
        assert clause.params == ["year", 2020]

    def test_a_string_ordering_does_not_cast(self):
        # Postgres can order text, unlike Qdrant. So `lt("name", "m")` is supported here
        # rather than refused — the algebra is the same, and the backends differ in what
        # they can honour.
        clause = store().compile(lt("name", "m"))

        assert "metadata ->> $1 < $2" in clause.text
        assert "::numeric" not in clause.text

    def test_in_is_an_array_membership(self):
        clause = store().compile(in_("tenant", ["acme", "globex"]))

        assert "jsonb_array_elements" in clause.text

    def test_placeholders_start_where_the_caller_says(self):
        # `search` has already used $1 for the query vector. A compiler that always
        # started at $1 would bind the filter's first value over the top of it — and the
        # query would still run, against a vector made of a tenant name.
        clause = store().compile(eq("tenant", "acme"), start=2)

        assert "$2" in clause.text
        assert "$1" not in clause.text

    def test_and_or_nest_with_parentheses(self):
        # Without them, `a AND b OR c` binds the wrong way and the filter silently means
        # something else.
        clause = store().compile((eq("a", 1) | eq("b", 2)) & eq("c", 3))

        assert clause.text.startswith("((")
        assert " OR " in clause.text
        assert " AND " in clause.text


class TestSearch:
    async def test_the_ordering_uses_the_collections_operator(self):
        pool = FakePool(rows=[])
        s = store(pool)
        await s.ensure_collection("docs", dimensions=2, distance=Distance.COSINE)

        await s.search("docs", VectorQuery(vector=[1.0, 0.0]))

        [select] = [q for q in sql_of(s) if q.startswith("SELECT id")]
        assert "vector <=> $1::vector" in select
        assert "ORDER BY vector <=> $1::vector" in select

    async def test_euclidean_uses_a_different_operator(self):
        pool = FakePool(rows=[])
        s = store(pool)
        await s.ensure_collection("docs", dimensions=2, distance=Distance.EUCLIDEAN)

        await s.search("docs", VectorQuery(vector=[1.0, 0.0]))

        [select] = [q for q in sql_of(s) if q.startswith("SELECT id")]
        assert "vector <-> $1::vector" in select

    async def test_the_filter_is_in_the_where_clause_not_applied_afterwards(self):
        pool = FakePool(rows=[])
        s = store(pool)
        await s.ensure_collection("docs", dimensions=2)

        await s.search("docs", VectorQuery(vector=[1.0, 0.0], where=eq("tenant", "acme")))

        [select] = [q for q in sql_of(s) if q.startswith("SELECT id")]
        assert "WHERE (metadata ? $2" in select

    async def test_ef_search_is_set_so_a_filtered_search_can_find_its_k(self):
        # An HNSW walk that runs out of budget before finding top_k *matching* rows
        # returns fewer results than it should, silently. This is the knob that fixes it,
        # and it is worthless if the adapter never sets it.
        pool = FakePool(rows=[])
        s = store(pool, ef_search=200)
        await s.ensure_collection("docs", dimensions=2)

        await s.search("docs", VectorQuery(vector=[1.0, 0.0]))

        assert any("hnsw.ef_search = 200" in q for q in sql_of(s))


class TestScores:
    @pytest.mark.parametrize(
        ("distance", "raw", "expected"),
        [
            # `<=>` is cosine *distance*: 0 is identical.
            (Distance.COSINE, 0.0, 1.0),
            (Distance.COSINE, 0.5, 0.5),
            # `<#>` is the **negative** inner product — a pgvector quirk, because it only
            # has ascending index scans and so negates the one metric where bigger is
            # better.
            (Distance.DOT, -0.8, 0.8),
            # `<->` is Euclidean distance.
            (Distance.EUCLIDEAN, 0.0, 1.0),
        ],
    )
    async def test_higher_is_always_better(self, distance, raw, expected):
        # All three operators would rank backwards if handed straight to a caller. The
        # search would work. It would return the least relevant documents first.
        pool = FakePool(rows=[{"id": "a", "metadata": "{}", "distance": raw}])
        s = store(pool)
        await s.ensure_collection("docs", dimensions=2, distance=distance)

        [hit] = await s.search("docs", VectorQuery(vector=[1.0, 0.0]))

        assert hit.score == pytest.approx(expected)


class TestWriting:
    async def test_upsert_replaces_on_conflict(self):
        s = store()

        await s.upsert("docs", [VectorRecord(id="a", vector=[1.0], metadata={"x": 1})])

        [statement] = [q for q in sql_of(s) if q.startswith("INSERT")]
        assert "ON CONFLICT (id) DO UPDATE" in statement

    async def test_a_large_write_is_batched(self):
        pool = FakePool()
        s = store(pool, batch_size=2)

        await s.upsert("docs", [VectorRecord(id=f"{i}", vector=[1.0]) for i in range(5)])

        writes = [args for text, args in pool.statements if text.startswith("INSERT")]
        assert [len(a) for a in writes] == [2, 2, 1]

    async def test_an_empty_write_touches_nothing(self):
        s = store()

        await s.upsert("docs", [])

        assert s._pool.statements == []

    async def test_delete_reports_what_postgres_actually_removed(self):
        # Unlike Qdrant, Postgres says. The command tag is exact rather than a count
        # taken beforehand and hoped to still be true.
        s = store(FakePool(status="DELETE 3"))

        assert await s.delete("docs", ids=["a", "b", "c"]) == 3

    async def test_delete_needs_exactly_one_selector(self):
        s = store()

        with pytest.raises(InvalidInputError):
            await s.delete("docs")


class TestCollections:
    async def test_it_creates_the_table_the_indexes_and_the_registry_row(self):
        s = store()

        await s.ensure_collection("docs", dimensions=384, distance=Distance.COSINE)

        statements = sql_of(s)
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in q for q in statements)
        assert any("vector vector(384)" in q for q in statements)
        # An HNSW index for the vectors, and a GIN index for the metadata — without the
        # latter, every filtered search is a sequential scan, and the filter that was
        # meant to make the query cheap makes it the most expensive thing in the system.
        assert any("USING hnsw (vector vector_cosine_ops)" in q for q in statements)
        assert any("USING gin (metadata)" in q for q in statements)
        assert any(q.startswith("INSERT INTO sarb_collections") for q in statements)

    async def test_the_metric_decides_the_index_operator_class(self):
        s = store()

        await s.ensure_collection("docs", dimensions=2, distance=Distance.EUCLIDEAN)

        assert any("vector_l2_ops" in q for q in sql_of(s))

    async def test_an_existing_collection_of_the_right_shape_is_left_alone(self):
        s = store(FakePool(row={"dimensions": 384, "distance": "cosine"}))

        await s.ensure_collection("docs", dimensions=384, distance=Distance.COSINE)

        assert not any(q.startswith('CREATE TABLE IF NOT EXISTS "sarb_docs"') for q in sql_of(s))

    async def test_a_reshape_is_refused(self):
        # There is nowhere else to remember the shape — in pgvector the metric lives in
        # the index opclass and the query operator, not in the column — so the registry
        # table is what makes this check possible at all.
        s = store(FakePool(row={"dimensions": 768, "distance": "cosine"}))

        with pytest.raises(VectorStoreError, match="cannot be reshaped"):
            await s.ensure_collection("docs", dimensions=384)


class TestPort:
    def test_it_satisfies_the_port_by_shape(self):
        assert isinstance(store(), VectorStore)

    async def test_closing_releases_the_pool(self):
        s = store()

        await s.aclose()

        assert s._pool.closed

    async def test_a_backend_failure_is_retryable(self):
        class Broken(FakePool):
            def acquire(self):
                raise RuntimeError("the connection pool is exhausted")

        with pytest.raises(VectorStoreError) as caught:
            await store(Broken()).count("docs")

        assert caught.value.retryable
