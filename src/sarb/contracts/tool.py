"""The tool port: what a model may call, and what it gets back.

A :class:`ToolProvider` is a *namespace* of tools rather than a single tool. That
is not incidental: it is the shape MCP servers, plugin bundles and remote tool
services already have, and modelling one tool per component would mean a config
block per tool — which stops being tolerable at about six.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from sarb.contracts.base import SarbModel
from sarb.contracts.component import Family
from sarb.contracts.message import ToolCall, ToolResult

__all__ = [
    "TOOLS",
    "ToolContext",
    "ToolProvider",
    "ToolSpec",
]


class ToolSpec(SarbModel):
    """What the model is told about a tool.

    Attributes:
        name: The identifier the model emits. Must match what the provider accepts:
            letters, digits, underscores and hyphens.
        description: What the tool does, written **for the model**. This is a
            prompt, not documentation — it is the single biggest lever on whether
            the tool gets called correctly, and it is worth more care than the
            implementation.
        parameters: A JSON Schema object describing the arguments.
        strict: Ask the provider to enforce the schema during decoding, where it
            can. Eliminates malformed arguments rather than recovering from them.
    """

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
    )
    strict: bool = False

    @model_validator(mode="after")
    def _parameters_must_be_an_object_schema(self) -> Self:
        """Providers require a top-level object schema; a bare type is rejected."""
        if self.parameters.get("type") != "object":
            raise ValueError(
                f"tool {self.name!r}: parameters must be a JSON Schema of type 'object'"
            )
        return self


class ToolContext(SarbModel):
    """Ambient facts a tool may need, that the model must not control.

    The separation matters. Arguments come from the *model* and are therefore
    untrusted input; context comes from the *application*. A tool that read the
    tenant id out of its arguments would let a model — and so, transitively, a user
    — choose whose data to read.

    Attributes:
        session_id: The conversation this call belongs to.
        principal: Who the call is being made on behalf of. Authorisation decisions
            belong here, never in arguments.
        metadata: Application-supplied extras.
    """

    session_id: str | None = None
    principal: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ToolProvider(Protocol):
    """A namespace of callable tools.

    One provider might wrap an MCP server, a set of Python functions, or a remote
    HTTP service.

    On failure: a tool that fails should return a :class:`~sarb.contracts.message.ToolResult`
    with ``is_error=True`` and a message the *model* can act on — not raise. A raise
    ends the agent's turn; an error result lets the model read "the date must be
    ISO-8601" and try again, which is what a competent colleague would do. Raise
    :class:`~sarb.errors.ToolError` only when the failure is the *framework's*, not
    the model's: the tool is missing, or the provider itself is broken.
    """

    async def list_tools(self) -> Sequence[ToolSpec]:
        """The tools this provider exposes.

        Called at least once per agent run to build the model's tool list, so it
        should be cheap — cache remote listings inside the adapter.
        """
        ...

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Run one tool call.

        Args:
            call: What the model asked for. ``call.arguments`` is **untrusted**: it
                is model output, and must be validated against the tool's schema.
                Note :attr:`~sarb.contracts.message.ToolCall.malformed_arguments`,
                which is set when the model emitted invalid JSON — return an error
                result, and it will get the chance to fix it.
            context: Application-supplied facts the model cannot influence.

        Returns:
            The result, with ``is_error=True`` if the tool failed in a way the model
            could recover from.

        Raises:
            ToolError: The tool does not exist, or the provider itself failed.
        """
        ...


#: The ``tools`` config family.
TOOLS = Family("tools", ToolProvider)
