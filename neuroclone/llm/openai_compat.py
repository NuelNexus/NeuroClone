"""Any OpenAI-compatible chat server: Ollama, LM Studio, llama.cpp, vLLM, OpenRouter, OpenAI."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, AsyncIterator, Optional

import aiohttp

from ..config import LLMConfig
from ..speech.text import ThinkFilter
from .base import JSON_INSTRUCTION, LLM, LLMError, Messages
from .jsonutil import JSONExtractError, extract_json

log = logging.getLogger(__name__)

_JSON_MODES = ("json_schema", "json_object", "prompt")


class _Unsupported(Exception):
    pass


class OpenAICompatLLM(LLM):
    name = "openai"
    supports_images = True

    def __init__(self, cfg: LLMConfig, session: Optional[aiohttp.ClientSession] = None) -> None:
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self.json_mode = cfg.json_mode

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(trust_env=True)
            self._owns_session = True
        return self._session

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _body(self, system: str, messages: Messages, max_tokens: Optional[int], stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if self.cfg.presence_penalty:
            body["presence_penalty"] = self.cfg.presence_penalty
        if self.cfg.frequency_penalty:
            body["frequency_penalty"] = self.cfg.frequency_penalty
        if self.cfg.seed is not None:
            body["seed"] = self.cfg.seed
        body.update(self.cfg.extra_body or {})
        return body

    async def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        body = self._body(system, messages, max_tokens, stream=True)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.cfg.timeout_s)
        think = ThinkFilter()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=timeout
            ) as resp:
                if resp.status != 200:
                    raise LLMError(f"{self.base_url} returned {resp.status}: {(await resp.text())[:300]}")
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise LLMError(f"stream error: {chunk['error']}")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        cleaned = think.feed(content)
                        if cleaned:
                            yield cleaned
                tail = think.flush()
                if tail:
                    yield tail
        except aiohttp.ClientError as exc:
            raise LLMError(f"cannot reach {self.base_url}: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"{self.base_url} timed out") from exc

    async def _post(self, body: dict) -> dict:
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
        try:
            async with session.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=timeout
            ) as resp:
                text = await resp.text()
                if resp.status in (400, 404, 415, 422, 501):
                    raise _Unsupported(f"{resp.status}: {text[:200]}")
                if resp.status != 200:
                    raise LLMError(f"{self.base_url} returned {resp.status}: {text[:300]}")
                return json.loads(text)
        except aiohttp.ClientError as exc:
            raise LLMError(f"cannot reach {self.base_url}: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"{self.base_url} timed out") from exc

    @staticmethod
    def _message_text(payload: dict) -> str:
        try:
            content = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion: {str(payload)[:200]}") from exc
        think = ThinkFilter()
        return think.feed(content) + think.flush()

    async def complete_json(
        self, system: str, messages: Messages, schema: dict, *, name: str = "output", max_tokens: Optional[int] = None
    ) -> Any:
        modes = _JSON_MODES if self.json_mode == "auto" else (self.json_mode,)
        errors = []
        for mode in modes:
            try:
                result = await self._json_call(mode, system, messages, schema, name, max_tokens)
            except (_Unsupported, JSONExtractError) as exc:
                errors.append(f"{mode}: {exc}")
                log.debug("JSON mode %s failed: %s", mode, exc)
                continue
            if self.json_mode == "auto" and mode != "json_schema":
                log.info("%s: using JSON mode %r from now on", self.base_url, mode)
                self.json_mode = mode
            return result
        raise LLMError("could not obtain JSON: " + " | ".join(errors))

    async def _json_call(
        self, mode: str, system: str, messages: Messages, schema: dict, name: str, max_tokens: Optional[int]
    ) -> Any:
        if mode == "prompt":
            try:
                return await self._prompted_json(system, messages, schema, max_tokens=max_tokens)
            except LLMError as exc:
                raise _Unsupported(str(exc)) from exc
        instruction = JSON_INSTRUCTION.format(schema=json.dumps(schema, ensure_ascii=False))
        body = self._body(f"{system}\n\n{instruction}", messages, max_tokens or 1024, stream=False)
        body["temperature"] = min(self.cfg.temperature, 0.7)
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        else:
            body["response_format"] = {"type": "json_object"}
        return extract_json(self._message_text(await self._post(body)))

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        b64 = base64.b64encode(image).decode("ascii")
        body = self._body(
            "You describe images for a livestreamer. Be concrete and brief.",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    ],
                }
            ],
            400,
            stream=False,
        )
        try:
            return self._message_text(await self._post(body)).strip()
        except _Unsupported as exc:
            raise LLMError(f"image input not supported by {self.cfg.model}: {exc}") from exc

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
