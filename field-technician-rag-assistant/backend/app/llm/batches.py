"""Message Batches: half price, results within 24 h. For work nobody waits on (nightly summaries).

One submit per night; the next night collects. `poll` returns None while the batch is still processing, else
{custom_id: text-or-None}, where None means that request failed and should simply be retried later.
"""
from __future__ import annotations

from typing import Protocol

import anthropic

from app.config import settings
from app.llm.base import LLMUnavailable


class BatchClient(Protocol):
    def submit(self, requests: list[dict]) -> str: ...

    def poll(self, batch_id: str) -> dict[str, str | None] | None: ...


class AnthropicBatches:
    def __init__(self, api_key: str | None = None):
        key = api_key or settings.claude_api_key.get_secret_value()
        if not key:
            raise LLMUnavailable("CLAUDE_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key, max_retries=settings.claude_max_retries)

    def submit(self, requests: list[dict]) -> str:
        return self._client.messages.batches.create(requests=requests).id

    def poll(self, batch_id: str) -> dict[str, str | None] | None:
        if self._client.messages.batches.retrieve(batch_id).processing_status != "ended":
            return None
        out: dict[str, str | None] = {}
        for r in self._client.messages.batches.results(batch_id):
            if r.result.type == "succeeded" and r.result.message.stop_reason not in ("refusal", "max_tokens"):
                out[r.custom_id] = next((b.text for b in r.result.message.content if b.type == "text"), None)
            else:
                out[r.custom_id] = None
        return out
