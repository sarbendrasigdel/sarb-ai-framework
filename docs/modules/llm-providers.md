# LLM providers

Five adapters ship in the box. They are the first real implementations of a port,
and so they are also the worked example every future adapter — embedders, vector
stores, rerankers — should be read against.

| `provider:` | Extra | What it talks to |
| --- | --- | --- |
| `anthropic` | `sarb[anthropic]` | Claude, via the Anthropic Messages API |
| `openai` | `sarb[openai]` | OpenAI Chat Completions — and any API-compatible server (vLLM, Groq, Together, LiteLLM) via `base_url` |
| `gemini` | `sarb[gemini]` | Gemini, via Google's Gen AI SDK; Vertex AI too |
| `ollama` | `sarb[ollama]` | A local Ollama daemon, over its native HTTP API |
| `fake` | — | A scripted model. No network, no key |

Selecting one is a line of YAML:

```yaml
llms:
  default:
    provider: anthropic
    options:
      model: claude-opus-4-8
      max_tokens: 4096
```

Your code never sees that line:

```python
llm = await container.get(LLMS)          # an LLMProvider. That is all it is.
response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))
```

## The shape of an adapter

Every one of the five is the same four pieces, in the same order.

```python
PROVIDER = "anthropic"
EXTRA = "anthropic"


class AnthropicOptions(LLMOptions):        # 1. the options schema
    api_key: SecretStr | None = None
    max_tokens: int = Field(default=4096, gt=0)
    thinking: bool = False


class AnthropicProvider:                   # 2. the adapter
    def __init__(self, client: AsyncAnthropic, options: AnthropicOptions) -> None:
        self._client = client
        self._options = options

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...


def build(ctx: ComponentContext) -> AnthropicProvider:   # 3. the factory
    sdk = import_optional("anthropic", extra=EXTRA, used_by=f"LLM provider {PROVIDER!r}")
    options = parse_options(ctx.config, AnthropicOptions, component=ctx.path)
    return AnthropicProvider(sdk.AsyncAnthropic(...), options)
```

…and 4., a line in `sarb/plugins/builtins.py` mapping `"anthropic"` to `build`.

Three properties of that sketch are load-bearing, and each one exists because the
obvious alternative hurts later.

### The kernel never learns the vendor's name

`AnthropicOptions` is declared **by the adapter**, not by the config module. The
config layer sees only `{provider: str, options: dict}` — an opaque bag — and hands
it to the factory, which validates it against its own schema through
`parse_options`. Nothing in `sarb.config` mentions `max_tokens` or `thinking`, and
a plugin nobody has written yet gets exactly the same treatment as Anthropic does.

The payoff is in the error message. Because `parse_options` is told the config path,
a typo comes back as the YAML key you typed:

```
ConfigurationError: llms.default.options.chunk_size: Input should be greater than 0
```

not as a Pydantic traceback about a class the operator has never heard of.

### The vendor client is injected, not constructed

`AnthropicProvider.__init__` takes an `AsyncAnthropic`. It does not build one. The
factory does that.

This is what makes translation testable. Every adapter's real work is turning our
`LLMRequest` into the vendor's request shape and the vendor's response back into our
`LLMResponse` — and with the client injected, all of that can be tested against the
vendor's *own* response models with no key, no network, and no mocking library:

```python
llm = AnthropicProvider(FakeClient(reply(text("Paris."))), options)

response = await llm.generate(LLMRequest(messages=(Message.user("hi"),)))

assert response.text == "Paris."
assert sent(llm)["system"] == "Be terse."   # what would have gone over the wire
```

The fixtures in `tests/unit/providers/llm/` are built from `anthropic.types.Message`,
`openai.types.chat.ChatCompletion`, `google.genai.types.Part` — the SDK validates the
shape we *claim* to receive. A hand-written dict can be wrong in precisely the same
way the adapter is wrong, and the two will then agree with each other, happily,
forever, while production fails.

### The SDK import happens inside `build`

Never at module level. `sarb.plugins.builtins` imports all five adapter modules on
every boot, so a top-level `import anthropic` would make `sarb[anthropic]` mandatory
for everyone.

`import_optional` turns the absence into an instruction:

```
MissingDependencyError: LLM provider 'anthropic' requires an optional dependency.
Install it with:  uv add "sarb[anthropic]"
```

All five are registered unconditionally, installed or not. A conditional
registration would make `sarb providers list` differ per machine, and would turn a
missing extra into "no such provider" — which sends you looking for a typo instead
of running one command.

## What the adapters share

`providers/llm/_shared.py` is small and carries most of the risk in the module: four
adapters delegate their trickiest behaviour to it, so a bug there is a bug in every
provider at once, including the ones nobody has written yet.

**`LLMOptions`** — the base every adapter's options extend. `model` has **no
default**: a framework that picks a model for you picks the wrong one and bills you
for it. `extra="forbid"`, so `temprature: 0.7` in YAML fails loudly rather than
leaving an operator convinced they set something.

**`ResponseBuilder`** — assembles a streamed response. Every adapter streams
differently (Anthropic's typed events, OpenAI's deltas, Gemini's parts, Ollama's
NDJSON) and every adapter must end with the *same* fully-assembled `LLMResponse` in
its `StreamEnd`. The builder keeps parallel tool calls in an index-keyed dict, so two
calls streamed interleaved cannot have their argument fragments concatenated into one
unparseable string — which is the bug this class exists to prevent.

**`parse_arguments`** — never raises. A model that emits `{"q":` produces a `ToolCall`
with `is_malformed=True` and `malformed_arguments` carrying the original text, and the
agent loop hands that back to the model to fix. Crashing on the model's typo would
make an ordinary, recoverable event into an outage.

**`error_for_status`** — one status-code table, one set of retry semantics. `429` →
`RateLimitError(retryable=True)` carrying `Retry-After`; `401`/`403` →
`AuthenticationError(retryable=False)`; `5xx` → retryable. A resilience policy written
against `sarb.errors` works for a provider it has never heard of.

**`unsupported`** — the escape hatch, and the module's firmest rule.

!!! danger "An adapter that cannot express a request raises. It never silently drops it."
    Ollama has no `tool_choice`. So `ToolChoice.REQUIRED` against Ollama raises
    `InvalidInputError`, naming the provider and the feature. It does not quietly send
    `auto` and let you discover the difference from a bill or a wrong answer.

    The exception is anything the port itself documents as best-effort — `seed`, for
    instance — which an adapter may drop, because the contract already said it might.

## Where the vendors actually differ

The four adapters are not four copies of one file. Each one is bent around something
specific to its API, and these are the parts worth reading before writing a fifth:

- **Anthropic** has no `system` role — it is a top-level parameter. `max_tokens` is
  mandatory, so the adapter always sends one. Thinking blocks come back with a
  **signature**, which must be echoed back byte-for-byte on the next turn or the API
  rejects it; an unsigned `ReasoningPart` is therefore dropped rather than sent. The
  stream is consumed as raw events, not through the SDK's accumulating helper, because
  `message_delta.usage.output_tokens` reports a running *total* and must replace rather
  than add.
- **OpenAI** is the only adapter whose message translation is not one-to-one: our single
  tool turn holding *n* results fans out into *n* separate `{"role": "tool"}` messages.
  An assistant turn made only of tool calls must send `content: None` — `""` is
  rejected. Malformed arguments are echoed back **unrepaired**, because the model needs
  to see what it actually produced. Chat Completions bills reasoning tokens but never
  returns them, so `capabilities.reasoning` is `False`.
- **Gemini** renames the assistant role to `model`, and reports `finish_reason: STOP`
  *even when it emits function calls* — the adapter corrects that to `TOOL_CALLS`, since
  an agent loop that trusted it would stop mid-task. Reasoning is distinguished from
  answer text only by a `thought: True` flag on a part.
- **Ollama** issues **no tool-call ids at all**. The adapter mints one, and matches
  results back by tool *name* on the way out. It uses raw `httpx` — the `ollama` extra
  is just `httpx` — so its tests run against a real `httpx.MockTransport` and assert the
  request that would genuinely have gone on the wire.

Each of those is a fact about a vendor. None of them leaked into `sarb.contracts`,
which is the point: the port describes what an LLM *is*, and the mess of what four
vendors happen to *do* is contained in four files.

## No model capability table

There is no `MODELS = {"claude-opus-4-8": {"context": 200_000, ...}}` anywhere in this
framework, and there will not be one.

Such a table is wrong the day a vendor ships a model, and it fails *closed* — refusing a
model that works. Adapters instead declare what the **API** supports (`capabilities`
describes streaming, tools, vision, JSON schema), and anything per-model is config:

```yaml
options:
  model: claude-opus-4-8
  max_input_tokens: 200000     # you know this. The framework should not guess.
```

Same reasoning that keeps price out of `Usage`.

## The fake is a provider, not a fixture

`provider: fake` is a real entry in the registry. Point a whole application at it and
it boots, serves, and streams — with no key and no network.

```yaml
llms:
  default:
    provider: fake
    options:
      responses: ["Paris."]
      tool_calls: [{ name: "search", arguments: { q: "capital" } }]
```

```python
llm = FakeProvider(FakeOptions(responses=("Paris.",)))

await llm.generate(request)

assert llm.requests[0].system == "Be terse."   # what the engine actually sent
```

That last assertion is usually the one that matters. Prompt-assembly bugs are silent: a
test that only checks the answer passes happily while the system prompt is being dropped
on the floor.

The fake can also be told to misbehave, which is the only honest way to test a retry
policy — the alternative is getting rate limited by a real vendor, deliberately, in CI:

```yaml
options:
  fail_with: rate_limit          # or auth, timeout, error
  tool_calls: [{ name: "search", malformed: '{"q":' }]
  latency: 0.2
```

Its `stream()` is implemented on top of its `generate()`, so the two paths cannot drift
apart. A fake whose two paths disagree finds bugs that do not exist and hides the ones
that do.

## Adding a provider

1. Create `src/sarb/providers/llm/<name>.py`. Copy the four-part shape above.
2. Extend `LLMOptions` with what your vendor needs. Nothing more — every knob is a thing
   someone will have to understand.
3. Implement `generate`, `stream`, `capabilities`, `aclose`. Lean on `ResponseBuilder`,
   `parse_arguments`, `error_for_status`. Raise `unsupported(...)` for anything the API
   cannot do.
4. Add the extra to `pyproject.toml` and register the key in `plugins/builtins.py`.
5. Write the tests from the **vendor's own** response models, with the client injected.

You never touch `sarb.config`, `sarb.di`, or `sarb.contracts`. If you find yourself
needing to, that is a real finding about the port — bring it up rather than working
around it.

## API reference

::: sarb.providers.llm
