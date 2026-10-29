"""Scripted LLM for tests: no network, fully deterministic. Every call is recorded in `.calls`."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from pydantic import BaseModel

from app.llm.base import LLMError, StructuredResult, TextStream, Usage

_APPROVE = {"verdict": "approve", "issues": [], "warnings": [], "rationale": "scripted approval"}


class FakeLLM:
    model = "fake-llm"

    def __init__(self, script: dict[str, list] | None = None, text: str | list[str] = "Fake answer."):
        # script[node] = list of BaseModel instances (or callables (user)->BaseModel), consumed in order;
        # the last item repeats so a single entry covers every call.
        # The safety review approves unless a test scripts it: every answer passes through it, and a
        # test about retrieval should not have to restate that the answer was safe.
        self.script: dict[str, list] = defaultdict(list, {"safety_review": [_APPROVE], **(script or {})})
        self.text = [text] if isinstance(text, str) else list(text)
        self.calls: list[dict] = []

    def structured(self, *, node, system, user, schema, effort=None, max_tokens=4000, fast=False, images=None):
        self.calls.append({"node": node, "system": system, "user": user, "images": len(images or [])})
        queue = self.script.get(node)
        if not queue:
            raise LLMError(f"FakeLLM has no script for node '{node}'")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        parsed = item(user) if isinstance(item, Callable) and not isinstance(item, BaseModel) else item
        if not isinstance(parsed, schema):
            parsed = schema.model_validate(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed)
        return StructuredResult(parsed=parsed, usage=Usage(10, 5), model=self.model, reasoning=f"[fake:{node}]")

    def stream_text(self, *, system, user, effort=None, max_tokens=3000, fast=False, images=None):
        self.calls.append({"node": "synthesize", "system": system, "user": user, "images": len(images or [])})
        if isinstance(self.text, str):  # tests may assign a plain string after construction
            self.text = [self.text]
        text = self.text.pop(0) if len(self.text) > 1 else self.text[0]
        handle = TextStream(iter(()))
        handle.model = self.model
        handle.usage = Usage(20, 15)
        # emit in small fragments to exercise the sentence-buffering guard
        handle._it = iter([text[i:i + 7] for i in range(0, len(text), 7)])
        return handle
