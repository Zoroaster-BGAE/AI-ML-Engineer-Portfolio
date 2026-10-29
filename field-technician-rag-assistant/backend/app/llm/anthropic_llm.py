"""Claude via the official Anthropic SDK. Structured nodes use schema-constrained output.

Every call goes through the beta surface because both options this app uses are betas there:
server-side refusal fallbacks (`fallbacks: "default"`) and fast mode (`speed: "fast"`).
"""
from __future__ import annotations

import base64

import anthropic
import pydantic

from app.config import settings
from app.llm.base import LLMError, LLMRefusal, LLMUnavailable, StructuredResult, TextStream, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FAST_BETA = "fast-mode-2026-02-01"


def _usage(u) -> Usage:
    return Usage(tokens_in=(u.input_tokens or 0), tokens_out=(u.output_tokens or 0))


def _content(user: str, images) -> str | list[dict]:
    """Photos first, then the text that refers to them."""
    if not images:
        return user
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": i.media_type,
                                           "data": base64.b64encode(i.data).decode("ascii")}} for i in images]
    return [*blocks, {"type": "text", "text": user}]


def _options(fast: bool) -> dict:
    betas, extra = [], {}
    if settings.claude_fallbacks:
        betas.append(FALLBACK_BETA)
        extra["fallbacks"] = "default"
    if fast and settings.claude_fast_mode:
        betas.append(FAST_BETA)
        extra["speed"] = "fast"
    return {"betas": betas, **extra} if betas else {}


class AnthropicLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or settings.claude_api_key.get_secret_value()
        if not key:
            raise LLMUnavailable("CLAUDE_API_KEY is not set")
        self.model = model or settings.claude_model
        self._client = anthropic.Anthropic(api_key=key, max_retries=settings.claude_max_retries,
                                           timeout=settings.claude_timeout_s)

    @staticmethod
    def _wrap(exc: Exception) -> LLMError:
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return LLMUnavailable(f"auth: {exc}")
        if isinstance(exc, anthropic.NotFoundError):
            return LLMUnavailable(f"model not found: {exc}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMUnavailable(f"connection: {exc}")
        return LLMError(f"{type(exc).__name__}: {exc}")

    def _parse(self, fast: bool, **kw):
        try:
            return self._client.beta.messages.parse(**kw, **_options(fast))
        except anthropic.RateLimitError:
            if not (fast and settings.claude_fast_mode):
                raise
            # Fast mode has its own, smaller rate limit; the standard lane usually has room.
            return self._client.beta.messages.parse(**kw, **_options(False))

    def structured(self, *, node, system, user, schema, effort=None, max_tokens=4000, fast=False, images=None):
        last: Exception | None = None
        for _ in range(2):  # a malformed / truncated generation gets exactly one retry
            try:
                resp = self._parse(
                    fast,
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": _content(user, images)}],
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": effort or settings.effort_nodes},
                    output_format=schema,
                )
            except anthropic.APIError as exc:
                raise self._wrap(exc) from exc
            except (pydantic.ValidationError, ValueError) as exc:  # invalid / truncated JSON from the model
                last = exc
                continue
            if resp.stop_reason == "refusal":
                raise LLMRefusal(f"refusal at node {node}")
            if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
                last = LLMError(f"node {node}: output truncated (stop_reason={resp.stop_reason})")
                continue
            reasoning = "\n".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking) or None
            return StructuredResult(parsed=resp.parsed_output, usage=_usage(resp.usage), model=resp.model,
                                    reasoning=reasoning)
        raise LLMError(f"node {node}: no valid structured output after 2 attempts: {str(last)[:300]}")

    def stream_text(self, *, system, user, effort=None, max_tokens=3000, fast=False, images=None):
        handle = TextStream(iter(()))
        handle.model = self.model
        kw = dict(model=self.model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": _content(user, images)}], thinking={"type": "adaptive"},
                  output_config={"effort": effort or settings.effort_synthesis})

        def gen():
            use_fast = fast
            while True:
                started = False
                try:
                    with self._client.beta.messages.stream(**kw, **_options(use_fast)) as stream:
                        for text in stream.text_stream:
                            started = True
                            yield text
                        final = stream.get_final_message()
                        handle.usage = _usage(final.usage)
                        handle.model = final.model
                        if final.stop_reason == "refusal":
                            raise LLMRefusal("refusal during synthesis")
                    return
                except anthropic.RateLimitError as exc:
                    if use_fast and settings.claude_fast_mode and not started:
                        use_fast = False
                        continue
                    raise self._wrap(exc) from exc
                except anthropic.APIError as exc:
                    raise self._wrap(exc) from exc

        handle._it = gen()
        return handle
