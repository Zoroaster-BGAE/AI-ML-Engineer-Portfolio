"""The only seam between the agent and any model provider. Swapping providers = one new file."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Any model failure. `user_message` is safe to show a technician; `str(exc)` is for logs/traces."""

    user_message = "The assistant could not reach its language model. Please try again in a moment."


class LLMUnavailable(LLMError):
    user_message = "The assistant's language model is unavailable right now (check the API key and network)."


class LLMRefusal(LLMError):
    user_message = "The model declined to answer this request. Try rephrasing the question."


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, other: Usage) -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out


@dataclass(frozen=True)
class ImageInput:
    """A photo sent with the question (panel, alarm display, nameplate), already re-encoded by app.media."""

    media_type: str
    data: bytes


@dataclass
class StructuredResult[T: BaseModel]:
    parsed: T
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    reasoning: str | None = None  # thinking summary, admin-only (agent_step.reasoning_raw)


class TextStream:
    """Iterable of text deltas; `.usage` is filled once the stream is exhausted."""

    def __init__(self, it: Iterator[str]):
        self._it = it
        self.usage = Usage()
        self.model = ""

    def __iter__(self) -> Iterator[str]:
        return self._it


class LLMClient(Protocol):
    model: str

    def structured(self, *, node: str, system: str, user: str, schema: type[T], effort: str | None = None,
                   max_tokens: int = 4000, fast: bool = False,
                   images: list[ImageInput] | None = None) -> StructuredResult[T]: ...

    def stream_text(self, *, system: str, user: str, effort: str | None = None,
                    max_tokens: int = 3000, fast: bool = False,
                    images: list[ImageInput] | None = None) -> TextStream: ...
