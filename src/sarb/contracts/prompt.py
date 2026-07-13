"""The prompt port: where prompts come from, and how they are versioned.

The port provides templates; it does not render them. Rendering is a pure function
of a template and some variables, and lives in the prompt *engine*. Keeping it out
of the port means a provider is only a store — filesystem, database, S3, a prompt
management SaaS — and swapping one does not change how templates are written.

Why versions are first class
----------------------------
A prompt is not code and is not data; it behaves like a **model weight**. Changing
one changes the system's output distribution, and the change cannot be reviewed by
reading the diff — only by evaluating it. So prompts need the things weights need:
an identity, a version, and the ability to answer "which prompt produced this
answer?" six weeks after the fact. A framework that treats prompts as f-strings in
source cannot answer that question, and the answer is exactly what you need during
an incident.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from sarb.contracts.base import Metadata, SarbModel
from sarb.contracts.component import Family

__all__ = ["PROMPTS", "PromptProvider", "PromptTemplate"]


class PromptTemplate(SarbModel):
    """A named, versioned prompt.

    Attributes:
        name: Stable identifier, e.g. ``"rag.answer"``.
        version: Which revision this is. Recorded on every run that uses it, so an
            answer can be traced back to the exact prompt that produced it.
        template: The template body, in whatever syntax the rendering engine takes.
        system: An optional system prompt shipped with it — the two are written
            together and drift apart if stored apart.
        variables: The variables it expects. Declared, so that a missing one is
            caught when the prompt is loaded rather than discovered as a literal
            ``{customer_name}`` in a message to a customer.
        metadata: Author, description, evaluation scores, whatever the team tracks.
    """

    name: str = Field(min_length=1)
    version: str = "1"
    template: str
    system: str | None = None
    variables: tuple[str, ...] = ()
    metadata: Metadata = Field(default_factory=dict)


@runtime_checkable
class PromptProvider(Protocol):
    """A source of prompt templates."""

    async def get(self, name: str, *, version: str | None = None) -> PromptTemplate:
        """Fetch a template.

        Args:
            name: The template's name.
            version: Which revision. ``None`` means the latest — convenient in
                development, and something a production deployment should pin, for
                the same reason it pins a model.

        Raises:
            PromptError: No such template, or no such version.
        """
        ...

    async def list(self) -> Sequence[PromptTemplate]:
        """Every template available. For the CLI, and for evaluation sweeps."""
        ...


#: The ``prompts`` config family.
PROMPTS = Family("prompts", PromptProvider)
