"""The LLM interface every backend implements."""

from __future__ import annotations

import abc
import json
import logging
from typing import Any, AsyncIterator, Optional

from .jsonutil import JSONExtractError, extract_json

log = logging.getLogger(__name__)

Messages = list[dict]  # [{"role": "user"|"assistant", "content": str}], user first, alternating

JSON_INSTRUCTION = (
    "Respond with ONLY a single JSON object that satisfies this JSON schema. "
    "No prose, no code fences.\nSchema:\n{schema}"
)


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    """The provider declined to answer (safety classifier). Treated like a filtered reply."""


class LLM(abc.ABC):
    """Streaming chat model. ``system`` must be stable across calls so prefix caches stay warm."""

    name: str = "llm"
    supports_images: bool = False

    @abc.abstractmethod
    def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        """Yield text deltas. ``purpose`` is "chat" for live speech, "utility" for background work."""

    async def complete(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "utility"
    ) -> str:
        parts = [delta async for delta in self.stream(system, messages, max_tokens=max_tokens, purpose=purpose)]
        return "".join(parts)

    async def complete_json(
        self,
        system: str,
        messages: Messages,
        schema: dict,
        *,
        name: str = "output",
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Return parsed JSON. Backends override this with native structured output when available."""
        return await self._prompted_json(system, messages, schema, max_tokens=max_tokens)

    async def _prompted_json(self, system: str, messages: Messages, schema: dict, *, max_tokens: Optional[int]) -> Any:
        instruction = JSON_INSTRUCTION.format(schema=json.dumps(schema, ensure_ascii=False))
        text = await self.complete(f"{system}\n\n{instruction}", messages, max_tokens=max_tokens, purpose="utility")
        try:
            return extract_json(text)
        except JSONExtractError as exc:
            raise LLMError(f"model did not return JSON: {exc}") from exc

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        raise LLMError(f"{self.name} does not support images")

    async def aclose(self) -> None:
        return None
