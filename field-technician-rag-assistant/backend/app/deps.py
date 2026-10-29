"""Process-wide singletons behind FastAPI dependencies, so tests can swap in FakeLLM / HashEmbedder."""
from functools import lru_cache

from app.llm.anthropic_llm import AnthropicLLM
from app.llm.base import LLMClient
from app.llm.embed import Embedder, get_embedder


@lru_cache
def _llm() -> LLMClient:
    return AnthropicLLM()


@lru_cache
def _embedder() -> Embedder:
    return get_embedder()


def get_llm() -> LLMClient:
    return _llm()


def get_embedder_dep() -> Embedder:
    return _embedder()
