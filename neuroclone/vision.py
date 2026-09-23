"""Optional screen vision: capture -> vision-language model -> silent context.

Neuro-sama's vision was reported at ~5 s latency. Here it never blocks speech: captures run in
a thread, descriptions arrive as background context, and the character mentions them only when
relevant. ``pip install 'neuroclone[vision]'`` and a model that accepts images.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Callable, Optional

from .config import VisionConfig
from .events import VisionObservation
from .llm.base import LLM

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "You are the eyes of an AI streamer. In at most two sentences, describe what is on this screen that "
    "a streamer would react to: the game and what's happening, any text or chat on screen, anything funny."
)


class VisionModule:
    def __init__(self, cfg: VisionConfig, llm: LLM, submit: Callable[[object], None]) -> None:
        self.cfg = cfg
        self.llm = llm
        self.submit = submit
        self._closing = False
        self._busy = asyncio.Lock()

    def capture(self) -> bytes:
        try:
            import mss
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("vision needs: pip install 'neuroclone[vision]'") from exc
        with mss.mss() as sct:
            monitor = sct.monitors[min(self.cfg.monitor, len(sct.monitors) - 1)]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if img.width > self.cfg.max_width:
            img = img.resize((self.cfg.max_width, int(img.height * self.cfg.max_width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    async def look(self, prompt: str = DEFAULT_PROMPT, image: Optional[bytes] = None) -> str:
        if self._busy.locked():
            return ""
        async with self._busy:
            try:
                data = image if image is not None else await asyncio.to_thread(self.capture)
                description = (await self.llm.describe_image(data, prompt)).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("vision failed: %s", exc)
                return ""
        if description:
            self.submit(VisionObservation(description))
        return description

    async def run(self) -> None:
        if self.cfg.interval_s <= 0:
            return
        while not self._closing:
            await self.look()
            await asyncio.sleep(self.cfg.interval_s)

    async def aclose(self) -> None:
        self._closing = True
