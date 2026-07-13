"""The workflow port: a DAG of steps, executed with observable state.

Status: this is the **least settled** port in the framework, and it is worth saying
so rather than projecting a confidence that has not been earned. The others are
constrained by an external reality — an LLM API, a vector database, a filesystem —
which sharply limits how wrong the shape can be. Workflow orchestration has no such
anchor: the design space runs from "a list of steps" to a full actor system, and the
right answer depends on requirements this framework has not met yet.

So the model here is deliberately the *smallest thing that is genuinely useful*: a
DAG of steps, each naming a component to invoke, with declared dependencies and a
state bag threaded through. Enough for document pipelines and multi-stage RAG, which
are the workflows people actually build first.

Two things it deliberately does **not** attempt, because getting them wrong is worse
than not having them:

* **Durable execution** — resuming a half-finished run after a process dies. That is
  a genuinely hard distributed-systems problem, and the honest options are to use
  something built for it or to accept that runs are in-memory. It is not something to
  half-implement.
* **Cycles and unbounded loops.** A DAG is acyclic. Iteration belongs in the agent
  engine, which has a turn limit and a model deciding when to stop.

When there is a real requirement that this shape cannot serve, it should be
*replaced*, not extended into a Turing machine one field at a time. The port is small
precisely so that replacing it stays cheap.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from sarb.contracts.base import SarbModel, new_id
from sarb.contracts.component import Family

__all__ = [
    "WORKFLOWS",
    "RunStatus",
    "Step",
    "StepFinished",
    "StepStarted",
    "Workflow",
    "WorkflowEvent",
    "WorkflowExecutor",
    "WorkflowFinished",
    "WorkflowRun",
]


class Step(SarbModel):
    """One node in a workflow.

    Attributes:
        id: Unique within the workflow. Referenced by ``needs`` and used as the key
            its output is stored under.
        uses: What to invoke, as ``family.component`` — e.g. ``retrievers.default``
            or ``tools.search``. A *reference*, resolved through the container at
            execution time, which is what keeps a workflow definition free of any
            vendor and safely storable as data.
        arguments: Arguments for the step. Values may reference earlier outputs;
            the engine defines the syntax. Written as ``with:`` in YAML, which is
            a Python keyword and so cannot be the attribute name.
        needs: Steps that must finish first. The edges of the DAG.
        when: An optional condition; the step is skipped when it is falsy.
    """

    # Spelled out rather than merged onto SarbModel.model_config: the pydantic mypy
    # plugin reads this statically, and cannot see through a dict-merge expression —
    # so a merged config silently leaves the plugin thinking `with` is the only
    # accepted argument name, and `Step(arguments=...)` fails to type check.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        populate_by_name=True,
    )

    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    uses: str = Field(pattern=r"^[a-z_]+\.[a-zA-Z0-9_-]+$")
    arguments: dict[str, Any] = Field(default_factory=dict, alias="with")
    needs: tuple[str, ...] = ()
    when: str | None = None


class Workflow(SarbModel):
    """A named, versioned DAG of steps.

    Data, not code — so it can be stored, diffed, versioned, and shipped without a
    deploy.
    """

    name: str = Field(min_length=1)
    version: str = "1"
    description: str = ""
    steps: tuple[Step, ...]

    @model_validator(mode="after")
    def _is_a_dag(self) -> Self:
        """Reject duplicate ids, dangling ``needs``, and cycles.

        Validated when the workflow is *loaded*, not when it runs. A cycle discovered
        half way through a pipeline has already spent real money on model calls, and
        may already have written to a real index.
        """
        if not self.steps:
            raise ValueError("a workflow needs at least one step")

        ids = [step.id for step in self.steps]
        duplicates = {name for name in ids if ids.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate step ids: {', '.join(sorted(duplicates))}")

        known = set(ids)
        for step in self.steps:
            missing = set(step.needs) - known
            if missing:
                raise ValueError(f"step {step.id!r} needs unknown step(s): {sorted(missing)}")

        self._reject_cycles()
        return self

    def _reject_cycles(self) -> None:
        """Depth-first search for a back edge, naming the cycle it finds."""
        edges = {step.id: step.needs for step in self.steps}
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str, path: tuple[str, ...]) -> None:
            if node in done:
                return
            if node in visiting:
                cycle = " -> ".join([*path[path.index(node) :], node])
                raise ValueError(f"workflow has a cycle: {cycle}")
            visiting.add(node)
            for parent in edges[node]:
                visit(parent, (*path, node))
            visiting.discard(node)
            done.add(node)

        for step in self.steps:
            visit(step.id, ())

    def order(self) -> tuple[tuple[str, ...], ...]:
        """Steps grouped into stages that may run concurrently.

        Each stage depends only on earlier ones, so an executor can run everything in
        a stage in parallel. Returning the *stages* rather than a flat topological
        order is what makes that parallelism available rather than merely possible.

        Returns:
            Stages, in execution order.

        Example:
            >>> workflow.order()
            (('load',), ('chunk', 'summarise'), ('index',))
        """
        remaining = {step.id: set(step.needs) for step in self.steps}
        stages: list[tuple[str, ...]] = []

        while remaining:
            ready = tuple(sorted(sid for sid, needs in remaining.items() if not needs))
            stages.append(ready)
            for sid in ready:
                del remaining[sid]
            for needs in remaining.values():
                needs.difference_update(ready)

        return tuple(stages)


class RunStatus(StrEnum):
    """Where a run got to."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkflowRun(SarbModel):
    """The outcome of executing a workflow.

    Attributes:
        id: This run.
        workflow: Name of the workflow.
        version: Version of the workflow — so a result can be traced to the exact
            definition that produced it.
        status: How it ended.
        outputs: Each completed step's output, keyed by step id.
        skipped: Steps whose ``when`` was falsy.
        error: Why it failed, if it did.
    """

    id: str = Field(default_factory=new_id)
    workflow: str
    version: str = "1"
    status: RunStatus = RunStatus.RUNNING
    outputs: dict[str, Any] = Field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    error: str | None = None


class StepStarted(SarbModel):
    """A step began."""

    kind: Literal["step_started"] = "step_started"
    step: str


class StepFinished(SarbModel):
    """A step ended.

    ``skipped`` distinguishes "its condition was false" from "it ran and produced
    nothing", which look identical in the outputs and mean very different things.
    """

    kind: Literal["step_finished"] = "step_finished"
    step: str
    output: Any = None
    skipped: bool = False
    error: str | None = None


class WorkflowFinished(SarbModel):
    """The run ended, carrying the full result."""

    kind: Literal["workflow_finished"] = "workflow_finished"
    run: WorkflowRun


#: One event from a running workflow. Discriminated on ``kind``.
type WorkflowEvent = Annotated[
    StepStarted | StepFinished | WorkflowFinished,
    Field(discriminator="kind"),
]


@runtime_checkable
class WorkflowExecutor(Protocol):
    """Runs workflows."""

    async def execute(self, workflow: Workflow, inputs: dict[str, Any]) -> WorkflowRun:
        """Run a workflow to completion.

        Raises:
            WorkflowError: The workflow could not be executed. A *step* failing is
                reported in the run's status and error, not raised — a partial run
                that recorded which steps succeeded is far more useful than a
                traceback that discards them.
        """
        ...

    def stream(self, workflow: Workflow, inputs: dict[str, Any]) -> AsyncIterator[WorkflowEvent]:
        """Run a workflow, reporting progress as it goes.

        The final event is always :class:`WorkflowFinished`.
        """
        ...


#: The ``workflows`` config family.
WORKFLOWS = Family("workflows", WorkflowExecutor)
