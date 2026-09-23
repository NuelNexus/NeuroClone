"""LLM backends and factory."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLM, LLMError, LLMRefusal, Messages


def create_llm(cfg: LLMConfig) -> LLM:
    provider = (cfg.provider or "ollama").lower()
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM(delay_s=cfg.mock_delay_s, seed=cfg.seed)
    if provider == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(cfg)
    if provider in ("openai", "openai_compat", "lmstudio", "llamacpp", "koboldcpp", "vllm", "openrouter"):
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(cfg)
    if provider in ("anthropic", "claude"):
        from .anthropic_backend import AnthropicLLM

        return AnthropicLLM(cfg)
    raise LLMError(f"unknown llm.provider {cfg.provider!r} (use ollama, openai, anthropic, or mock)")


__all__ = ["LLM", "LLMError", "LLMRefusal", "Messages", "create_llm"]
