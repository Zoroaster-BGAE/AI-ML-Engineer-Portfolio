"""Embedders. Claude has no embeddings endpoint, so production uses bge-m3 on local Ollama."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from app.config import settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingUnavailable(RuntimeError):
    pass


class FastEmbedder:
    """In-process ONNX embeddings (fastembed). The model is downloaded once to FASTEMBED_CACHE_PATH."""

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model_name = model or settings.embed_model
        self.dim = dim or settings.embed_dim
        self._model = None

    def _m(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return [v.tolist() for v in self._m().embed(texts, batch_size=32)]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailable(f"fastembed failed ({self.model_name}): {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OllamaEmbedder:
    def __init__(self, base_url: str | None = None, model: str | None = None, dim: int | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embed_model
        self.dim = dim or settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), 16):
                r = httpx.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts[i:i + 16]},
                               timeout=300)
                r.raise_for_status()
                out.extend(r.json()["embeddings"])
        except (httpx.HTTPError, KeyError) as exc:
            raise EmbeddingUnavailable(f"Ollama embeddings failed ({self.model} @ {self.base_url}): {exc}") from exc
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbedder:
    """Deterministic bag-of-words hashing into `dim` dims. Texts sharing words land close together,
    which is enough to exercise retrieval, isolation and ranking in tests without any model."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def get_embedder() -> Embedder:
    return {"hash": HashEmbedder, "ollama": OllamaEmbedder}.get(settings.embedder, FastEmbedder)()
