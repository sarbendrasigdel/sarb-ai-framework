# Conventions

The mechanical rules are enforced by `ruff` and `mypy`, and you should not have to
think about them: run `ruff format` and move on. This page is about the things a
linter cannot check, which are the things reviewers will actually raise.

## The rules that are really about architecture

These look like style. They are not.

**Never import a vendor SDK at module level.** Not in an adapter, not "just for a type
hint", not under `if TYPE_CHECKING` if you can avoid it. `plugins/builtins.py` imports
every adapter module on every boot, so a top-level `import anthropic` makes
`sarb[anthropic]` mandatory for everyone who installs `sarb`. The import belongs inside
`build()`, behind `import_optional`.

**Never construct a vendor client inside an adapter.** It is injected. The factory
builds it. This is what lets the adapter be tested against the vendor's own response
models with no key and no network.

**Never put a vendor's name in the kernel.** No `OpenAIConfig | AnthropicConfig` union,
no `if provider == "anthropic"` anywhere outside `providers/`. Adding a provider must
not require editing a core file. If it does, the extension mechanism is broken and
that is the bug to fix.

**Never let an engine import a provider.** It is a test, not an aspiration —
see [testing](testing.md#the-architecture-tests).

## Types

`mypy --strict`, and it is not negotiable. `disallow_any_generics` is on: `dict` is an
error, `dict[str, int]` is not.

`Any` is permitted in exactly three places, each with a written-down reason in
`pyproject.toml`:

- the DI container, which is heterogeneous by nature and re-narrows through overloads,
- structured logging, which takes arbitrary user values by definition,
- **the vendor boundary inside `providers/`** — annotating those with the vendor's own
  classes would force every SDK to be installed to type-check, which is the exact
  coupling the extras exist to prevent.

Everything crossing *back* into the framework is a validated model from
`sarb.contracts`, and that is where the type safety lives. If you want a fourth
exemption, that is a design conversation, not a `# type: ignore`.

## Errors

Every error the framework raises inherits `SarbError` and carries structured
`context`. An error message is a user interface — often the only one a stranger will
ever see — and it has one job: **tell them what to do next.**

```
MissingDependencyError: LLM provider 'anthropic' requires an optional dependency.
Install it with:  uv add "sarb[anthropic]"
```

```
ConfigurationError: llms.default.options.chunk_size: Input should be greater than 0
```

```
ProviderNotFoundError: No provider 'anthorpic' in family 'llms'.
Available: anthropic, fake, gemini, ollama, openai
```

Each of those names the thing that is wrong, in the vocabulary the user typed it in,
and either gives the command or lists the alternatives. Compare with what the naive
implementation produces — a `ModuleNotFoundError` from four frames down, or a Pydantic
traceback about a class the operator has never heard of. That difference is most of the
work in this framework's error paths, and it is worth the effort every time.

Two more rules:

- **Fail loudly, at the boundary.** An unknown option in YAML is an error, not a
  warning. A silently ignored `temprature: 0.7` leaves an operator convinced they set
  something.
- **Never silently degrade.** If an adapter cannot express a request, it raises. See
  `unsupported()` in `providers/llm/_shared.py`. The exception is anything the port
  explicitly documents as best-effort — `seed`, for instance — because the contract
  already said it might be dropped.

## Comments and docstrings

Public API gets a docstring; Google style; `ruff` checks it.

Comments are for the **why**, and specifically for the why that the code cannot show.
The bar: *would the next reader delete this line, believing it redundant, and break
something?* If yes, write the comment. If no, do not.

```python
# Anthropic reports output_tokens as a running total, not a delta, so this
# replaces rather than accumulates.
usage.output_tokens = event.usage.output_tokens
```

That is a comment. This is not:

```python
# Set the output tokens
usage.output_tokens = event.usage.output_tokens
```

Comments that describe what the next line does, or that narrate the history of a
change, are noise the moment the PR merges. `git blame` already knows.

The same standard applies to tests — a comment naming the bug a test prevents is worth
more than the test's own body:

```python
# The bug this exists to prevent: two calls streamed interleaved, their argument
# fragments concatenated into one unparseable string.
```

## Naming

- Test names are claims: `test_a_tool_turn_becomes_tool_result_blocks_in_one_user_message`,
  not `test_tools_2`. When it fails at 3am, the name is the bug report.
- Ports are nouns describing a capability — `LLMProvider`, `VectorStore`,
  `ChunkingStrategy` — never a vendor and never an implementation detail.
- Config keys are the vocabulary of the person writing YAML, not of the person writing
  Python.

## Documentation

Docs live in `docs/`, they build with `--strict`, and a broken link fails CI.

Every module gets a page, and each page explains **why the module is shaped this way**,
not merely what it does. An API reference is generated from docstrings by
`mkdocstrings` and needs no help from you; what it cannot generate is the reasoning,
and the reasoning is the part that stops the next person undoing your decision by
accident.

If you make a non-obvious call, write down what you rejected and why. That paragraph is
the most valuable thing in the file six months later.

## Commits and pull requests

- Commit messages: imperative mood, explaining *why*. `Send max_tokens on every
  Anthropic request` beats `fix bug`.
- Keep the diff to one concern. A provider and a refactor of the retry logic are two
  PRs.
- Green before review: `ruff`, `mypy`, `pytest`, `mkdocs build --strict`. All four run
  locally in under a minute, and there is no reason for a reviewer to be the one who
  discovers a formatting error.
