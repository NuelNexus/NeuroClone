"""Claude via the official ``anthropic`` SDK (``pip install 'neuroclone[anthropic]'``).

Live replies stream with low effort so speech starts fast; structured decisions use
``output_config.format`` (JSON schema). The persona system prompt is a cached block, and
server-side refusal fallbacks are enabled by default (``llm.fallbacks: false`` turns them off).
Sampling parameters and assistant prefill are not used (current models reject them).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, AsyncIterator, Optional

from ..config import LLMConfig
from .base import LLM, LLMError, LLMRefusal, Messages
from .jsonutil import JSONExtractError, extract_json

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LATENCY_HINT = "Latency-sensitive; begin your visible answer immediately."
_OLLAMA_DEFAULT = "http://localhost:11434/v1"


class AnthropicLLM(LLM):
    name = "anthropic"
    supports_images = True

    def __init__(self, cfg: LLMConfig, client: Any = None) -> None:
        self.cfg = cfg
        self.model = cfg.model if cfg.model and cfg.model.startswith("claude") else DEFAULT_MODEL
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("the anthropic provider needs: pip install 'neuroclone[anthropic]'") from exc
            kwargs: dict[str, Any] = {"max_retries": 2, "timeout": max(cfg.timeout_s, 30.0)}
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.base_url and cfg.base_url != _OLLAMA_DEFAULT:
                kwargs["base_url"] = cfg.base_url
            client = anthropic.AsyncAnthropic(**kwargs)
        self.client = client

    def _request(self, system: str, messages: Messages, *, max_tokens: int, effort: str, latency_hint: bool) -> dict:
        text = f"{system}\n\n{LATENCY_HINT}" if latency_hint else system
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": effort},
        }
        if self.cfg.fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def _error(self, exc: Exception) -> LLMError:
        try:
            import anthropic
        except ImportError:  # pragma: no cover - only reachable with an injected client
            return LLMError(str(exc))
        if isinstance(exc, anthropic.AuthenticationError):
            return LLMError("Anthropic rejected the API key (set ANTHROPIC_API_KEY or llm.api_key)")
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError("Anthropic rate limit hit; slow down or raise your limits")
        if isinstance(exc, anthropic.APIStatusError):
            return LLMError(f"Anthropic API error {exc.status_code}: {exc.message}")
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError("cannot reach the Anthropic API")
        return LLMError(str(exc))

    async def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        chat = purpose == "chat"
        request = self._request(
            system,
            messages,
            max_tokens=max_tokens or (4096 if chat else 8192),
            effort=self.cfg.effort if chat else self.cfg.json_effort,
            latency_hint=chat,
        )
        try:
            async with self.client.beta.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into LLMError
            raise self._error(exc) from exc
        if getattr(final, "stop_reason", None) == "refusal":
            details = getattr(final, "stop_details", None)
            raise LLMRefusal(f"declined ({getattr(details, 'category', None) or 'policy'})")

    async def complete_json(
        self, system: str, messages: Messages, schema: dict, *, name: str = "output", max_tokens: Optional[int] = None
    ) -> Any:
        request = self._request(
            system, messages, max_tokens=max_tokens or 4096, effort=self.cfg.json_effort, latency_hint=False
        )
        request["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        try:
            async with self.client.beta.messages.stream(**request) as stream:
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001
            raise self._error(exc) from exc
        if final.stop_reason == "refusal":
            raise LLMRefusal("declined a structured request")
        text = "".join(block.text for block in final.content if getattr(block, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return extract_json(text)
            except JSONExtractError as exc:
                raise LLMError(f"invalid JSON from Claude: {exc}") from exc

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(image).decode()},
            },
            {"type": "text", "text": prompt},
        ]
        request = self._request(
            "You describe images for a livestreamer. Be concrete and brief.",
            [{"role": "user", "content": content}],
            max_tokens=2048,
            effort="low",
            latency_hint=True,
        )
        try:
            async with self.client.beta.messages.stream(**request) as stream:
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001
            raise self._error(exc) from exc
        if final.stop_reason == "refusal":
            raise LLMRefusal("declined to describe the image")
        return "".join(b.text for b in final.content if getattr(b, "type", "") == "text").strip()

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass
