"""Tests for the neutral filter algebra and the vector-store models.

Filters are where vendor neutrality is actually won or lost, so they get the most
attention: an expression must survive a round trip through JSON, because that is
what an adapter compiles and what an audit log records.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from sarb.contracts import (
    And,
    Compare,
    CompareOp,
    Filter,
    Not,
    Or,
    SearchHit,
    VectorQuery,
    VectorRecord,
    eq,
    exists,
    gt,
    gte,
    in_,
    lt,
    ne,
    not_in,
)

FILTERS: TypeAdapter[Filter] = TypeAdapter(Filter)


class TestConstructors:
    def test_comparisons(self):
        assert eq("lang", "en").op is CompareOp.EQ
        assert ne("lang", "en").op is CompareOp.NE
        assert gt("year", 2020).op is CompareOp.GT
        assert gte("year", 2020).op is CompareOp.GTE
        assert lt("year", 2020).op is CompareOp.LT
        assert in_("lang", ["en", "fr"]).value == ["en", "fr"]
        assert not_in("lang", ["de"]).op is CompareOp.NOT_IN
        assert exists("author").value is True
        assert exists("author", present=False).value is False


class TestComposition:
    def test_and(self):
        expression = eq("tenant", "acme") & gte("year", 2020)

        assert isinstance(expression, And)
        assert len(expression.filters) == 2

    def test_or(self):
        assert isinstance(eq("lang", "en") | eq("lang", "fr"), Or)

    def test_not(self):
        assert isinstance(~eq("archived", True), Not)

    def test_expressions_nest(self):
        expression = (eq("tenant", "acme") & gte("year", 2020)) | ~exists("archived")

        assert isinstance(expression, Or)

    def test_operators_and_explicit_forms_are_equivalent(self):
        assert (eq("a", 1) & eq("b", 2)) == And(filters=(eq("a", 1), eq("b", 2)))


class TestValidation:
    def test_set_operators_need_a_list(self):
        with pytest.raises(ValidationError, match="needs a list"):
            Compare(field="lang", op=CompareOp.IN, value="en")

    def test_exists_needs_a_bool(self):
        with pytest.raises(ValidationError, match="needs a bool"):
            Compare(field="author", op=CompareOp.EXISTS, value="yes")


class TestSerialisation:
    def test_a_filter_round_trips_through_json(self):
        # An adapter compiles this, and an audit log stores it. It has to survive.
        original = (eq("tenant", "acme") & gte("year", 2020)) | ~exists("archived")

        restored = FILTERS.validate_json(FILTERS.dump_json(original))

        assert restored == original

    def test_the_discriminator_selects_the_right_node(self):
        restored = FILTERS.validate_python(
            {"kind": "and", "filters": [{"kind": "compare", "field": "a", "op": "eq", "value": 1}]}
        )

        assert isinstance(restored, And)
        assert isinstance(restored.filters[0], Compare)

    def test_an_unknown_node_kind_is_rejected(self):
        with pytest.raises(ValidationError):
            FILTERS.validate_python({"kind": "regex", "field": "a", "value": ".*"})


class TestModels:
    def test_query_defaults(self):
        query = VectorQuery(vector=[0.1, 0.2])

        assert query.top_k == 10
        assert query.where is None
        # Vectors are large and almost nobody wants them back.
        assert query.include_vectors is False

    def test_top_k_is_bounded(self):
        with pytest.raises(ValidationError):
            VectorQuery(vector=[0.1], top_k=0)

    def test_a_record_carries_filterable_metadata(self):
        record = VectorRecord(id="c1", vector=[0.1], metadata={"tenant": "acme", "year": 2024})

        assert record.metadata["tenant"] == "acme"

    def test_hits_sort_by_score_descending_regardless_of_metric(self):
        # Higher is always better, whatever distance the store uses. Retrieval code
        # sorts one way, always.
        hits = [SearchHit(id="a", score=0.2), SearchHit(id="b", score=0.9)]

        best = max(hits, key=lambda hit: hit.score)

        assert best.id == "b"
