"""Ollama's native API: the fastest way to run the brain on your own PC.

Why not Ollama's OpenAI-compatible endpoint? Only the native API lets us set, per request:

- ``num_ctx``: Ollama defaults to a 4k context on GPUs under 24 GB and silently drops the start of
  longer prompts, which is where the character's personality lives.
- ``think: false``: thinking models (Qwen 3.5, Gemma 4...) otherwise reason for seconds before the
  first word, which is dead air on stream.
- ``keep_alive``: keeps the model loaded between replies instead of unloading after 5 minutes.
- ``num_gpu: 0``: runs a helper model on the CPU so it never evicts the chat model from VRAM.
- ``format: <JSON schema>``: grammar-constrained structured output for game moves and memory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator, Optional

import aiohttp

from ..config import LLMConfig
from ..speech.text import ThinkFilter
from .base import JSON_INSTRUCTION, LLM, LLMError, Messages
from .jsonutil import JSONExtractError, extract_json

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
_JSON_MODES = ("schema", "json", "prompt")
# Options that make Ollama reload the model when they change between requests.
LOAD_OPTIONS = ("num_ctx", "num_gpu", "num_thread", "num_batch", "main_gpu", "use_mmap")


def native_url(base_url: str) -> str:
    """Accept the OpenAI-style URL too: ``http://host:11434/v1`` -> ``http://host:11434``."""
    url = (base_url or DEFAULT_URL).rstrip("/")
    for suffix in ("/v1", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def keep_alive_value(value: Any) -> Any:
    """Ollama takes a duration string ("30m") or seconds; negative means forever."""
    if isinstance(value, (int, float)):
        return value
    text = str(value or "").strip()
    try:
        return int(text)
    except ValueError:
        return text or "30m"


class _Unsupported(Exception):
    pass


class OllamaLLM(LLM):
    name = "ollama"
    supports_images = True

    def __init__(self, cfg: LLMConfig, session: Optional[aiohttp.ClientSession] = None) -> None:
        self.cfg = cfg
        self.base_url = native_url(cfg.base_url)
        self._session = session
        self._owns_session = session is None
        self.json_mode = "schema" if cfg.json_mode in ("auto", "json_schema") else cfg.json_mode
        if self.json_mode not in _JSON_MODES:
            self.json_mode = "schema"
        self.think: Optional[bool] = cfg.think
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ plumbing
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Ollama is local: never route it through an HTTP proxy from the environment.
            self._session = aiohttp.ClientSession(trust_env=False)
            self._owns_session = True
        return self._session

    def options(self, max_tokens: Optional[int] = None, *, temperature: Optional[float] = None) -> dict:
        cfg = self.cfg
        opts: dict[str, Any] = {
            "num_ctx": cfg.num_ctx,
            "temperature": cfg.temperature if temperature is None else temperature,
            "top_p": cfg.top_p,
            "num_predict": max_tokens or cfg.max_tokens,
        }
        if cfg.presence_penalty:
            opts["presence_penalty"] = cfg.presence_penalty
        if cfg.frequency_penalty:
            opts["frequency_penalty"] = cfg.frequency_penalty
        if cfg.num_gpu is not None:
            opts["num_gpu"] = cfg.num_gpu
        if cfg.num_thread:
            opts["num_thread"] = cfg.num_thread
        if cfg.seed is not None:
            opts["seed"] = cfg.seed
        opts.update(cfg.options or {})
        return opts

    def load_options(self) -> dict:
        """The subset of options that decides how the model is loaded (must match every request)."""
        return {k: v for k, v in self.options().items() if k in LOAD_OPTIONS}

    @staticmethod
    def _messages(system: str, messages: Messages) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):  # OpenAI-style parts -> text + base64 images
                texts, images = [], []
                for part in content:
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        images.append(url.split(",", 1)[1] if url.startswith("data:") else url)
                item = {"role": msg.get("role", "user"), "content": "\n".join(texts)}
                if images:
                    item["images"] = images
                out.append(item)
            else:
                out.append({"role": msg.get("role", "user"), "content": str(content)})
        return out

    def _body(self, system: str, messages: Messages, *, stream: bool, max_tokens: Optional[int] = None,
              temperature: Optional[float] = None) -> dict:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._messages(system, messages),
            "stream": stream,
            "options": self.options(max_tokens, temperature=temperature),
            "keep_alive": keep_alive_value(self.cfg.keep_alive),
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def _error(self, status: int, text: str) -> LLMError:
        try:
            detail = json.loads(text).get("error", text)
        except (ValueError, AttributeError):
            detail = text
        detail = str(detail)[:300]
        if status == 404 or "not found" in detail.lower():
            return LLMError(f"Ollama has no model {self.cfg.model!r}: run `ollama pull {self.cfg.model}` ({detail})")
        return LLMError(f"Ollama returned {status}: {detail}")

    def _unreachable(self, exc: Exception) -> LLMError:
        return LLMError(f"cannot reach Ollama at {self.base_url} ({exc}). Start the Ollama app (or `ollama serve`).")

    # ------------------------------------------------------------------ chat
    async def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        body = self._body(system, messages, stream=True, max_tokens=max_tokens)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.cfg.timeout_s)
        think = ThinkFilter()  # some model templates still put <think> blocks in the content
        try:
            async with session.post(f"{self.base_url}/api/chat", json=body, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if resp.status == 400 and "think" in text.lower() and self.think is not None:
                        # This model or Ollama version rejects the think flag: drop it for good and retry.
                        log.info("Ollama rejected the think flag (%s); retrying without it", text[:120])
                        self.think = None
                    else:
                        raise self._error(resp.status, text)
                else:
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise LLMError(f"Ollama stream error: {chunk['error']}")
                        content = (chunk.get("message") or {}).get("content")
                        if content:
                            cleaned = think.feed(content)
                            if cleaned:
                                yield cleaned
                        if chunk.get("done"):
                            self.last_stats = _stats(chunk)
                            break
                    tail = think.flush()
                    if tail:
                        yield tail
                    return
        except aiohttp.ClientError as exc:
            raise self._unreachable(exc) from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"Ollama at {self.base_url} stopped responding") from exc
        async for delta in self.stream(system, messages, max_tokens=max_tokens, purpose=purpose):
            yield delta  # only reached after dropping the think flag

    async def _chat_once(self, body: dict) -> str:
        session = await self._get_session()
        timeout = aiohttp.ClientTimeout(total=max(self.cfg.timeout_s, 30.0) * 2, sock_connect=10)
        try:
            async with session.post(f"{self.base_url}/api/chat", json=body, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status == 400 and "format" in text.lower():
                    raise _Unsupported(text[:200])
                if resp.status != 200:
                    raise self._error(resp.status, text)
                payload = json.loads(text)
        except aiohttp.ClientError as exc:
            raise self._unreachable(exc) from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"Ollama at {self.base_url} timed out") from exc
        self.last_stats = _stats(payload)
        content = (payload.get("message") or {}).get("content") or ""
        think = ThinkFilter()
        return think.feed(content) + think.flush()

    async def complete_json(
        self, system: str, messages: Messages, schema: dict, *, name: str = "output", max_tokens: Optional[int] = None
    ) -> Any:
        """Grammar-constrained JSON (``format: schema``); falls back to JSON mode, then to prompting."""
        start = _JSON_MODES.index(self.json_mode)
        errors = []
        for mode in _JSON_MODES[start:]:
            instruction = JSON_INSTRUCTION.format(schema=json.dumps(schema, ensure_ascii=False))
            body = self._body(f"{system}\n\n{instruction}", messages, stream=False, max_tokens=max_tokens or 1024,
                              temperature=min(self.cfg.temperature, 0.7))
            if mode == "schema":
                body["format"] = schema
            elif mode == "json":
                body["format"] = "json"
            try:
                result = extract_json(await self._chat_once(body))
            except (_Unsupported, JSONExtractError) as exc:
                errors.append(f"{mode}: {exc}")
                continue
            if mode != self.json_mode:
                log.info("Ollama %s: using JSON mode %r from now on", self.cfg.model, mode)
                self.json_mode = mode
            return result
        raise LLMError("could not obtain JSON: " + " | ".join(errors))

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        body = self._body("You describe images for a livestreamer. Be concrete and brief.",
                          [{"role": "user", "content": prompt}], stream=False, max_tokens=300)
        body["messages"][-1]["images"] = [base64.b64encode(image).decode("ascii")]
        try:
            return (await self._chat_once(body)).strip()
        except LLMError as exc:
            if "image" in str(exc).lower() or "vision" in str(exc).lower():
                raise LLMError(f"{self.cfg.model} cannot see images; pick a vision model such as qwen3.5 or "
                               f"gemma4 ({exc})") from exc
            raise

    # ------------------------------------------------------------------ model management
    async def _get_json(self, path: str, timeout: float = 5.0) -> Any:
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}{path}", timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    raise self._error(resp.status, await resp.text())
                return await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise self._unreachable(exc) from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"Ollama at {self.base_url} timed out") from exc

    async def version(self) -> str:
        return str((await self._get_json("/api/version")).get("version", "?"))

    async def installed(self) -> list[str]:
        data = await self._get_json("/api/tags")
        return [m.get("name") or m.get("model") for m in data.get("models", [])]

    async def has_model(self) -> bool:
        names = await self.installed()
        want = self.cfg.model if ":" in self.cfg.model else f"{self.cfg.model}:latest"
        return want in names or self.cfg.model in names

    async def residency(self) -> Optional[dict]:
        """How the loaded model is split between GPU and CPU (from /api/ps), or None if not loaded."""
        data = await self._get_json("/api/ps")
        for m in data.get("models", []):
            if m.get("name") in (self.cfg.model, f"{self.cfg.model}:latest") or m.get("model") == self.cfg.model:
                size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
                return {"size_gb": size / 1e9, "vram_gb": vram / 1e9,
                        "gpu_pct": round(100 * vram / size) if size else 0, "context": m.get("context_length")}
        return None

    async def warmup(self) -> float:
        """Load the model into memory now (with the same load options as every later request)."""
        session = await self._get_session()
        body = {"model": self.cfg.model, "messages": [], "keep_alive": keep_alive_value(self.cfg.keep_alive),
                "options": self.load_options()}
        start = time.monotonic()
        try:
            async with session.post(f"{self.base_url}/api/chat", json=body,
                                    timeout=aiohttp.ClientTimeout(total=600, sock_connect=10)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise self._error(resp.status, text)
        except aiohttp.ClientError as exc:
            raise self._unreachable(exc) from exc
        except asyncio.TimeoutError as exc:
            raise LLMError(f"loading {self.cfg.model} took over 10 minutes") from exc
        return time.monotonic() - start

    async def pull(self, progress=None) -> None:
        """Download the model (``ollama pull``), reporting (status, completed_bytes, total_bytes)."""
        session = await self._get_session()
        try:
            async with session.post(f"{self.base_url}/api/pull", json={"model": self.cfg.model, "stream": True},
                                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)) as resp:
                if resp.status != 200:
                    raise self._error(resp.status, await resp.text())
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("error"):
                        raise LLMError(f"pulling {self.cfg.model} failed: {item['error']}")
                    if progress is not None:
                        progress(item.get("status", ""), int(item.get("completed") or 0), int(item.get("total") or 0))
        except aiohttp.ClientError as exc:
            raise self._unreachable(exc) from exc

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()


def _stats(payload: dict) -> dict:
    """Speed numbers from a final Ollama response (durations are in nanoseconds)."""
    out: dict[str, float] = {}
    if payload.get("eval_count") and payload.get("eval_duration"):
        out["tokens_per_s"] = payload["eval_count"] / (payload["eval_duration"] / 1e9)
    if payload.get("prompt_eval_count") and payload.get("prompt_eval_duration"):
        out["prompt_tokens"] = payload["prompt_eval_count"]
        out["prompt_tokens_per_s"] = payload["prompt_eval_count"] / (payload["prompt_eval_duration"] / 1e9)
    if payload.get("load_duration"):
        out["load_s"] = payload["load_duration"] / 1e9
    return out
