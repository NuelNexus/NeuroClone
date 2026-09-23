"""Live speech first: sharing one GPU between the live reply and background LLM work.

On a single consumer GPU every LLM call competes for the same card. A memory summary that starts
a moment before a viewer's question would delay the answer by seconds. ``LivePriority`` fixes that:

- background calls (memory, reflections, summaries, vision) wait until no live reply is being
  generated, plus a short quiet period;
- when a live reply starts, running background calls are cancelled (which stops generation on
  the server) and retried later, so the viewer never waits on housekeeping;
- after a few preemptions a background call is allowed to finish, so it can't starve forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .base import LLM, Messages

log = logging.getLogger(__name__)


class LivePriority:
    def __init__(self, background_delay_s: float = 0.5, max_preemptions: int = 4,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.background_delay_s = background_delay_s
        self.max_preemptions = max_preemptions
        self.clock = clock
        self._live = 0
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._last_live_end = -1e9
        self._preemptible: set[asyncio.Task] = set()
        self.preempted = 0

    @property
    def live_active(self) -> bool:
        return self._live > 0

    @contextlib.asynccontextmanager
    async def live(self):
        self._live += 1
        self._quiet.clear()
        for task in list(self._preemptible):
            task.cancel()  # the background call is retried once things are quiet again
        try:
            yield
        finally:
            self._live -= 1
            if self._live == 0:
                self._last_live_end = self.clock()
                self._quiet.set()

    async def _wait_quiet(self) -> None:
        while True:
            await self._quiet.wait()
            remaining = self._last_live_end + self.background_delay_s - self.clock()
            if self._live == 0 and remaining <= 0:
                return
            await asyncio.sleep(min(max(remaining, 0.01), 1.0))

    async def run_background(self, make_call: Callable[[], Awaitable[Any]]) -> Any:
        attempts = 0
        while True:
            await self._wait_quiet()
            task = asyncio.ensure_future(make_call())
            protected = attempts >= self.max_preemptions
            if not protected:
                self._preemptible.add(task)
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:  # our caller is going away (e.g. shutdown)
                task.cancel()
                raise
            finally:
                self._preemptible.discard(task)
            if task.cancelled():
                attempts += 1
                self.preempted += 1
                log.debug("background LLM call paused for a live reply (attempt %d)", attempts)
                continue
            return task.result()


class PrioritizedLLM(LLM):
    """A view of an LLM whose calls are either live (hold the GPU) or background (yield to live)."""

    def __init__(self, inner: LLM, gate: LivePriority, *, live: bool) -> None:
        self.inner = inner
        self.gate = gate
        self.is_live = live
        self.name = inner.name
        self.supports_images = inner.supports_images

    async def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        if self.is_live:
            async with self.gate.live():
                async for delta in self.inner.stream(system, messages, max_tokens=max_tokens, purpose=purpose):
                    yield delta
            return
        text = await self.gate.run_background(
            lambda: self.inner.complete(system, messages, max_tokens=max_tokens, purpose=purpose))
        if text:
            yield text

    async def complete_json(self, system: str, messages: Messages, schema: dict, *, name: str = "output",
                            max_tokens: Optional[int] = None) -> Any:
        if self.is_live:
            async with self.gate.live():
                return await self.inner.complete_json(system, messages, schema, name=name, max_tokens=max_tokens)
        return await self.gate.run_background(
            lambda: self.inner.complete_json(system, messages, schema, name=name, max_tokens=max_tokens))

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        if self.is_live:
            async with self.gate.live():
                return await self.inner.describe_image(image, prompt, media_type)
        return await self.gate.run_background(lambda: self.inner.describe_image(image, prompt, media_type))

    async def aclose(self) -> None:
        return None  # the runtime closes the shared inner LLM once

    def __getattr__(self, item: str) -> Any:  # warmup(), residency(), cfg... of the wrapped backend
        return getattr(self.inner, item)
