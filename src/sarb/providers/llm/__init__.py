"""LLM adapters: one per vendor, all satisfying :class:`~sarb.contracts.LLMProvider`.

| ``provider:`` | Extra | Wraps |
| --- | --- | --- |
| ``anthropic`` | ``sarb[anthropic]`` | Claude, via the Messages API |
| ``openai`` | ``sarb[openai]`` | Chat Completions — and anything that speaks it |
| ``gemini`` | ``sarb[gemini]`` | Gemini, via the Gen AI SDK |
| ``ollama`` | ``sarb[ollama]`` | A local Ollama daemon, over plain HTTP |
| ``fake`` | — | A scripted model, for tests and for working offline |

Importing this module is safe and cheap on a core-only install: it pulls in no
vendor SDK, because every adapter defers its import into ``build()``. Configuring
a provider whose extra is missing therefore fails at *construction*, with the
install command in the message, rather than at import with a
``ModuleNotFoundError`` from somewhere nobody was looking.
"""

from __future__ import annotations

from sarb.providers.llm.anthropic import AnthropicOptions, AnthropicProvider
from sarb.providers.llm.fake import FakeOptions, FakeProvider, ScriptedToolCall
from sarb.providers.llm.gemini import GeminiOptions, GeminiProvider
from sarb.providers.llm.ollama import OllamaOptions, OllamaProvider
from sarb.providers.llm.openai import OpenAIOptions, OpenAIProvider

__all__ = [
    "AnthropicOptions",
    "AnthropicProvider",
    "FakeOptions",
    "FakeProvider",
    "GeminiOptions",
    "GeminiProvider",
    "OllamaOptions",
    "OllamaProvider",
    "OpenAIOptions",
    "OpenAIProvider",
    "ScriptedToolCall",
]
