"""Tests for the machinery every embedding adapter shares.

``check_output`` is the reason this file exists. The two failures it catches — a
misaligned batch and a mis-sized vector — do not raise anywhere else in the stack. A
vector store accepts both without complaint, and the only symptom is that retrieval
is quietly worse than it should be, for as long as the index lives. There is no
traceback to find later, so the check has to happen here, on every call.
"""

from __future__ import annotations

import math
import threading
import time

import anyio
import pytest

from sarb.errors import EmbeddingError
from sarb.providers.embedding._shared import (
    EmbeddingOptions,
    ThreadRunner,
    batches,
    check_output,
    normalized,
)


class TestBatches:
    def test_a_large_call_is_split(self):
        # The caller embeds a document, which is however many chunks it turned out to
        # be. The vendor's cap on inputs per request is not their problem.
        assert [list(b) for b in batches(["a", "b", "c", "d", "e"], 2)] == [
            ["a", "b"],
            ["c", "d"],
            ["e"],
        ]

    def test_a_small_call_is_one_batch(self):
        assert [list(b) for b in batches(["a"], 64)] == [["a"]]

    def test_nothing_is_no_batches(self):
        assert list(batches([], 64)) == []


class TestNormalized:
    def test_a_vector_becomes_unit_length(self):
        assert math.isclose(sum(v * v for v in normalized([3.0, 4.0])), 1.0)

    def test_direction_is_preserved(self):
        assert normalized([3.0, 4.0]) == [0.6, 0.8]

    def test_normalising_twice_changes_nothing(self):
        # Which is what makes it safe to apply uniformly, over models that already
        # normalise (most of fastembed's) and models that do not (most of torch's).
        once = normalized([1.0, 2.0, 3.0])

        assert normalized(once) == pytest.approx(once)

    def test_the_zero_vector_is_not_a_division_by_zero(self):
        # A model can legitimately emit it for unusual input. It has no direction to
        # preserve, and it is not worth an exception.
        assert normalized([0.0, 0.0]) == [0.0, 0.0]


class TestCheckOutput:
    def test_the_happy_path_returns_the_vectors(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]

        assert check_output(vectors, ["a", "b"], provider="p", model="m", dimensions=2) is vectors

    def test_too_few_vectors_is_an_error_and_not_a_shrug(self):
        # The failure this exists for: a provider drops one input, every vector after
        # it shifts up by one, and every chunk in the index is now findable only by
        # the text of the *next* chunk. Nothing downstream would notice.
        with pytest.raises(EmbeddingError, match="1 vectors for 2 texts"):
            check_output([[1.0, 2.0]], ["a", "b"], provider="p", model="m", dimensions=2)

    def test_the_wrong_width_is_an_error(self):
        # What a model change that was not carried through to the collection looks
        # like — and what a hosted API ignoring the `dimensions` it was sent looks
        # like. Both are one bad deploy away, and both are silent without this.
        with pytest.raises(EmbeddingError, match="3-dimensional vector where 2 was configured"):
            check_output([[1.0, 2.0, 3.0]], ["a"], provider="p", model="m", dimensions=2)

    def test_the_error_says_which_vector_was_wrong(self):
        with pytest.raises(EmbeddingError) as caught:
            check_output([[1.0, 2.0], [1.0]], ["a", "b"], provider="p", model="m", dimensions=2)

        assert caught.value.context["index"] == 1

    def test_embedding_nothing_is_not_a_failure(self):
        assert check_output([], [], provider="p", model="m", dimensions=2) == []


class TestThreadRunner:
    async def test_it_runs_the_work(self):
        assert await ThreadRunner(1).run(lambda: 6 * 7) == 42

    async def test_it_runs_off_the_event_loop(self):
        # The whole reason it exists. A torch forward pass on the loop thread stalls
        # every other request in the process for as long as it takes.
        loop_thread = threading.current_thread().ident

        assert await ThreadRunner(1).run(lambda: threading.current_thread().ident) != loop_thread

    async def test_it_does_not_run_more_at_once_than_it_was_allowed(self):
        # Unbounded `to_thread` is how you turn one slow model into an out-of-memory:
        # fifty concurrent requests, fifty batches, one set of weights. Queuing is the
        # correct behaviour, it is not what `to_thread` does by default, so it is
        # asserted rather than assumed.
        runner = ThreadRunner(2)
        lock = threading.Lock()
        concurrent = 0
        peak = 0

        def work() -> None:
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.02)
            with lock:
                concurrent -= 1

        async with anyio.create_task_group() as group:
            for _ in range(8):
                group.start_soon(runner.run, work)

        assert peak == 2


class TestEmbeddingOptions:
    def test_a_model_must_be_named(self):
        # Changing it invalidates every vector already stored. Not a value anything
        # should pick on your behalf.
        with pytest.raises(ValueError, match="model"):
            EmbeddingOptions()  # type: ignore[call-arg]

    def test_unknown_options_are_rejected(self):
        with pytest.raises(ValueError, match="dimension"):
            EmbeddingOptions(model="m", dimension=384)  # type: ignore[call-arg]

    def test_a_batch_size_of_zero_would_embed_nothing_forever(self):
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingOptions(model="m", batch_size=0)
