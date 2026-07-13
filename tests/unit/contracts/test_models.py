"""Tests for documents, chunks, tools, and workflow structure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sarb.contracts import (
    Chunk,
    Document,
    Step,
    ToolContext,
    ToolSpec,
    Workflow,
)


class TestChunkProvenance:
    def test_offsets_make_a_citation_verifiable(self):
        chunk = Chunk(document_id="d1", text="Paris", index=0, start_char=100, end_char=105)

        assert chunk.start_char == 100
        assert chunk.end_char == 105

    def test_half_an_offset_pair_is_refused(self):
        with pytest.raises(ValidationError, match="both start_char and end_char"):
            Chunk(document_id="d1", text="x", index=0, start_char=10)

    def test_an_inverted_span_is_refused(self):
        # A silently inverted span cites nonsense.
        with pytest.raises(ValidationError, match="must not precede"):
            Chunk(document_id="d1", text="x", index=0, start_char=50, end_char=10)

    def test_offsets_may_be_absent(self):
        assert Chunk(document_id="d1", text="x", index=0).start_char is None

    def test_a_document_gets_an_id_by_default(self):
        assert Document(content="hello").id


class TestToolSpec:
    def test_parameters_must_be_an_object_schema(self):
        # Providers reject a bare type; catching it here beats a 400 at runtime.
        with pytest.raises(ValidationError, match="type 'object'"):
            ToolSpec(name="search", description="x", parameters={"type": "string"})

    def test_a_name_the_provider_would_reject_is_refused(self):
        with pytest.raises(ValidationError):
            ToolSpec(name="search the web!", description="x")

    def test_a_description_is_required(self):
        # It is a prompt, not documentation, and the biggest lever on correct calls.
        with pytest.raises(ValidationError):
            ToolSpec(name="search", description="")

    def test_default_parameters_are_a_valid_empty_object(self):
        assert ToolSpec(name="now", description="Current time.").parameters["type"] == "object"


class TestToolContext:
    def test_the_principal_is_not_model_controlled(self):
        # Arguments come from the model and are untrusted; context comes from the
        # application. A tool reading its tenant from arguments would let a user
        # pick whose data to read.
        context = ToolContext(session_id="s1", principal="user-42")

        assert context.principal == "user-42"


class TestWorkflowStructure:
    def test_a_valid_dag(self):
        workflow = Workflow(
            name="ingest",
            steps=(
                Step(id="load", uses="loaders.default"),
                Step(id="chunk", uses="chunkers.default", needs=("load",)),
                Step(id="summarise", uses="llms.fast", needs=("load",)),
                Step(id="index", uses="vector_stores.default", needs=("chunk", "summarise")),
            ),
        )

        # Stages, not a flat order: an executor can run each stage concurrently.
        assert workflow.order() == (("load",), ("chunk", "summarise"), ("index",))

    def test_a_cycle_is_refused_at_load_not_at_run(self):
        # A cycle found half way through a pipeline has already spent real money
        # and may already have written to a real index.
        with pytest.raises(ValidationError, match="cycle"):
            Workflow(
                name="loop",
                steps=(
                    Step(id="a", uses="tools.x", needs=("b",)),
                    Step(id="b", uses="tools.x", needs=("a",)),
                ),
            )

    def test_a_self_referencing_step_is_a_cycle(self):
        with pytest.raises(ValidationError, match="cycle"):
            Workflow(name="w", steps=(Step(id="a", uses="tools.x", needs=("a",)),))

    def test_a_dangling_dependency_is_refused(self):
        with pytest.raises(ValidationError, match="unknown step"):
            Workflow(name="w", steps=(Step(id="a", uses="tools.x", needs=("ghost",)),))

    def test_duplicate_step_ids_are_refused(self):
        with pytest.raises(ValidationError, match="duplicate step ids"):
            Workflow(
                name="w",
                steps=(Step(id="a", uses="tools.x"), Step(id="a", uses="tools.y")),
            )

    def test_an_empty_workflow_is_refused(self):
        with pytest.raises(ValidationError, match="at least one step"):
            Workflow(name="w", steps=())

    def test_uses_must_name_a_family_and_a_component(self):
        with pytest.raises(ValidationError):
            Step(id="a", uses="just-a-tool-name")

    def test_a_workflow_is_data_and_round_trips(self):
        # It can be stored, diffed, versioned, and shipped without a deploy.
        original = Workflow(
            name="ingest",
            steps=(Step(id="load", uses="loaders.default", arguments={"path": "/x"}),),
        )

        restored = Workflow.model_validate_json(original.model_dump_json(by_alias=True))

        assert restored == original

    def test_arguments_are_written_as_with_in_yaml(self):
        step = Step.model_validate({"id": "a", "uses": "tools.x", "with": {"q": "hi"}})

        assert step.arguments == {"q": "hi"}
        assert step.model_dump(by_alias=True)["with"] == {"q": "hi"}
