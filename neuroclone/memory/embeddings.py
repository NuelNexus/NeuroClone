"""Text embedders. The hashing embedder needs nothing but numpy and works fully offline."""

from __future__ import annotations

import abc
import hashlib
import logging
import re
from typing import Optional

import aiohttp
import numpy as np

log = logging.getLogger(__name__)

_STOP = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its "
    "this that these those i me my you your he him his she her they them their we us our do does did "
    "so just very really too can could would should will shall have has had not no yes ok okay lol".split()
)


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 2 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


class Embedder(abc.ABC):
    dim: int = 0

    @property
    def signature(self) -> str:
        return f"{type(self).__name__}:{self.dim}"

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array with L2-normalised rows."""

    async def aclose(self) -> None:
        return None


class HashingEmbedder(Embedder):
    """Feature-hashed bag of stemmed words + bigrams. Deterministic and dependency-free."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def vector(self, text: str) -> np.ndarray:
        toks = [_stem(t) for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP]
        feats = [(t, 1.0) for t in toks] + [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, weight in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.vector(t) for t in texts])


class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible /embeddings endpoint (Ollama, LM Studio, llama.cpp, OpenAI...)."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.dim = 0
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def signature(self) -> str:
        return f"openai:{self.model}"

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(trust_env=True)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with self._session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"embeddings endpoint returned {resp.status}: {(await resp.text())[:200]}")
            payload = await resp.json()
        rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
        mat = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
        self.dim = mat.shape[1]
        return mat

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class OllamaEmbedder(Embedder):
    """Ollama's native /api/embed. ``on_cpu`` keeps the embedding model off the GPU (it is tiny and
    fast on a CPU), so it never competes with the chat model for VRAM."""

    def __init__(self, base_url: str, model: str, *, on_cpu: bool = True, keep_alive: str = "30m",
                 timeout_s: float = 30.0) -> None:
        from ..llm.ollama import keep_alive_value, native_url

        self.base_url = native_url(base_url)
        self.model = model
        self.options = {"num_gpu": 0} if on_cpu else {}
        self.keep_alive = keep_alive_value(keep_alive)
        self.timeout_s = timeout_s
        self.dim = 0
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def signature(self) -> str:
        return f"ollama:{self.model}"

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(trust_env=False)
        body = {"model": self.model, "input": texts, "keep_alive": self.keep_alive}
        if self.options:
            body["options"] = self.options
        async with self._session.post(f"{self.base_url}/api/embed", json=body,
                                      timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as resp:
            if resp.status != 200:
                text = (await resp.text())[:200]
                hint = f" (run `ollama pull {self.model}`)" if resp.status == 404 or "not found" in text else ""
                raise RuntimeError(f"Ollama embeddings returned {resp.status}: {text}{hint}")
            payload = await resp.json()
        mat = np.asarray(payload["embeddings"], dtype=np.float32)
        mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
        self.dim = mat.shape[1]
        return mat

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def create_embedder(kind: str, base_url: str = "", model: str = "", api_key: str = "", *,
                    on_cpu: bool = True) -> Embedder:
    if kind == "ollama":
        return OllamaEmbedder(base_url, model or "nomic-embed-text", on_cpu=on_cpu)
    if kind == "openai":
        return OpenAIEmbedder(base_url, model, api_key)
    if kind != "hash":
        log.warning("unknown memory.embedder %r, using the hashing embedder", kind)
    return HashingEmbedder()


def embedder_for(memory_cfg, llm_cfg) -> Embedder:
    """Build the configured embedder, defaulting its server to the chat model's server."""
    base = memory_cfg.embed_base_url or llm_cfg.base_url
    if memory_cfg.embedder == "openai" and not memory_cfg.embed_base_url and llm_cfg.provider == "ollama":
        from ..llm.ollama import native_url

        base = native_url(base) + "/v1"  # Ollama's OpenAI-compatible endpoints live under /v1
    return create_embedder(memory_cfg.embedder, base, memory_cfg.embed_model,
                           memory_cfg.embed_api_key or llm_cfg.api_key, on_cpu=memory_cfg.embed_on_cpu)
