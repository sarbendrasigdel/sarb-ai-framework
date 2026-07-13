"""The base every domain model shares.

Two settings carry real weight.

**Frozen.** A ``Message`` handed to three concurrent tool calls must be the same
message in all three. Immutability is what makes it safe to pass domain objects
across tasks without defensive copies, and it removes an entire category of bug
where a retry mutates the request it is retrying.

**extra="forbid".** A typo'd field on a model we construct ourselves should fail
where it is written, not silently vanish and produce a subtly wrong request.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["JSONValue", "Metadata", "SarbModel", "new_id"]

#: Anything that survives a round-trip through JSON. Used for metadata and filter
#: values, which cross process boundaries and must therefore be serialisable.
type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]

#: Free-form annotations carried by documents, chunks, and search results.
#:
#: Typed loosely on purpose: metadata is where an *application's* domain lives —
#: tenant, department, effective date — and the framework cannot know what that
#: is. Vector-store filters operate on these keys.
type Metadata = dict[str, JSONValue]


class SarbModel(BaseModel):
    """Base for every domain model in the framework."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        # Vendor payloads arrive as JSON; accepting a plain str where an enum is
        # expected keeps adapters from writing a coercion layer per field.
        use_enum_values=False,
        validate_default=True,
    )

    def evolve(self, **changes: Any) -> Any:
        """Return a copy with these fields replaced.

        The frozen-model equivalent of mutation, and deliberately more visible
        than assignment: producing a changed request should look like producing a
        new one, because that is what it is.

        Example:
            >>> retried = request.evolve(temperature=0.0)
        """
        return self.model_copy(update=changes)


def new_id() -> str:
    """A fresh opaque identifier for a document, chunk, or run.

    A UUID4 hex string. Not sequential, so it leaks no ordering or volume
    information, and needs no coordination to generate.
    """
    return uuid.uuid4().hex
