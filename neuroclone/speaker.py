"""The response pipeline: LLM token stream -> sentences -> guards -> TTS -> audio + lip-sync.

A producer turns tokens into sentences and starts synthesis early (bounded lookahead) while
a consumer plays finished clips in order. The first sentence plays while later ones are still
being generated. Filtering is per sentence: a blocked sentence is replaced by an in-character
deflection and the reply ends there. Interrupts:

- ``soon``: finish the current sentence plus at most one more that is already queued
- ``after_sentence``: finish the current sentence, drop the rest
- ``now``: stop audio immediately
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

from .emotion import EmotionEngine
from .events import EventBus
from .llm.base import LLMError, LLMRefusal
from .persona import Persona
from .repetition import RepetitionGuard
from .safety.filter import LLMModerator, OutputFilter, OutputVerdict
from .speech.audio import AudioClip, AudioPlayer
from .speech.text import SentenceChunker, clean_for_tts, extract_tags, has_speakable_content, strip_speaker_prefix
from .speech.tts import TTS

log = logging.getLogger(__name__)

_INTERRUPT_RANK = {None: 0, "soon": 1, "after_sentence": 2, "now": 3}
Hook = Callable[..., Union[None, Awaitable[None]]]


@dataclass
class SpokenResult:
    speaker: str
    spoken: list = field(default_factory=list)
    generated: str = ""
    emotions: list = field(default_factory=list)
    filtered: bool = False
    filter_reason: str = ""
    interrupted: bool = False
    interrupt_reason: str = ""
    skipped_repetitive: int = 0
    error: str = ""
    t_request: float = 0.0
    t_first_token: Optional[float] = None
    t_first_audio: Optional[float] = None
    t_end: float = 0.0
    enqueued: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.spoken).strip()

    def latency(self) -> dict:
        def rel(t: Optional[float]) -> Optional[float]:
            return None if t is None else round(t - self.t_request, 3)

        return {"first_token_s": rel(self.t_first_token), "first_audio_s": rel(self.t_first_audio),
                "total_s": rel(self.t_end)}


@dataclass
class _Utterance:
    text: str
    tags: list
    synth: asyncio.Task
    moderation: Optional[asyncio.Task] = None
    deflection: bool = False


@dataclass
class SpeakerHooks:
    on_sentence: Optional[Hook] = None  # (speaker, text, tags, clip)
    on_speaking: Optional[Hook] = None  # (speaker, is_speaking: bool)
    on_speech_finished: Optional[Hook] = None  # (speaker, is_final, cancelled, reason)
    on_level: Optional[Callable[[float], None]] = None  # lip-sync level 0..1
    on_expression: Optional[Hook] = None  # (emotion)


async def _call(hook: Optional[Hook], *args: Any) -> None:
    if hook is None:
        return
    try:
        result = hook(*args)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - hooks must never break speech
        log.exception("speaker hook failed")


async def _single(text: str) -> AsyncIterator[str]:
    yield text


class Speaker:
    def __init__(
        self,
        persona: Persona,
        tts: TTS,
        player: AudioPlayer,
        output_filter: OutputFilter,
        *,
        emotion: Optional[EmotionEngine] = None,
        repetition: Optional[RepetitionGuard] = None,
        bus: Optional[EventBus] = None,
        moderator: Optional[LLMModerator] = None,
        hooks: Optional[SpeakerHooks] = None,
        max_sentences: int = 4,
        first_clause_chars: int = 48,
        lookahead: int = 2,
        synth_timeout_s: float = 20.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.persona = persona
        self.name = persona.name
        self.tts = tts
        self.player = player
        self.output_filter = output_filter
        self.emotion = emotion or EmotionEngine()
        self.repetition = repetition or RepetitionGuard()
        self.bus = bus or EventBus()
        self.moderator = moderator
        self.hooks = hooks or SpeakerHooks()
        self.max_sentences = max_sentences
        self.first_clause_chars = first_clause_chars
        self.lookahead = lookahead
        self.synth_timeout_s = synth_timeout_s
        self.rng = rng or random.Random()
        self._interrupt: Optional[str] = None
        self._interrupt_reason = ""
        self._speaking = False
        self._carry_tags: list[str] = []
        self._last_deflection = ""
        self._queue: Optional[asyncio.Queue] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ public
    @property
    def speaking(self) -> bool:
        return self._speaking

    def interrupt(self, mode: str = "now", reason: str = "interrupted") -> None:
        if not self._speaking:
            return
        if _INTERRUPT_RANK.get(mode, 0) > _INTERRUPT_RANK.get(self._interrupt, 0):
            self._interrupt = mode
            self._interrupt_reason = reason
            if mode == "now":
                self.player.stop()
            if mode in ("now", "after_sentence") and self._queue is not None:
                self._queue.put_nowait(None)  # wake the consumer if it is waiting on the model

    async def speak_text(self, text: str, turn_id: str = "", t_request: Optional[float] = None) -> SpokenResult:
        return await self.speak_stream(_single(text), turn_id, t_request=t_request)

    async def speak_stream(
        self, tokens: AsyncIterator[str], turn_id: str = "", t_request: Optional[float] = None
    ) -> SpokenResult:
        async with self._lock:
            return await self._speak(tokens, turn_id, t_request or time.monotonic())

    # ------------------------------------------------------------------ pipeline
    async def _speak(self, tokens: AsyncIterator[str], turn_id: str, t_request: float) -> SpokenResult:
        result = SpokenResult(speaker=self.name, t_request=t_request)
        self._interrupt, self._interrupt_reason, self._carry_tags = None, "", []
        self._speaking = True
        queue: asyncio.Queue = asyncio.Queue()
        self._queue = queue
        slots = asyncio.Semaphore(self.lookahead)
        await _call(self.hooks.on_speaking, self.name, True)
        producer = asyncio.ensure_future(self._produce(tokens, queue, result, slots))
        try:
            await self._consume(queue, result, slots)
        finally:
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            while not queue.empty():
                item = queue.get_nowait()
                if item is not None:
                    item.synth.cancel()
                    if item.moderation:
                        item.moderation.cancel()
            if self._interrupt and not result.interrupted:
                result.interrupted = True
            result.interrupt_reason = self._interrupt_reason if result.interrupted else ""
            result.t_end = time.monotonic()
            self._speaking = False
            self._interrupt = None
            self._queue = None
            await _call(self.hooks.on_speech_finished, self.name, True, result.interrupted,
                        result.interrupt_reason or None)
            await _call(self.hooks.on_speaking, self.name, False)
            self.bus.publish("speech_end", {
                "speaker": self.name, "turn": turn_id, "text": result.text, "filtered": result.filtered,
                "interrupted": result.interrupted, "latency": result.latency(),
            })
        return result

    async def _produce(self, tokens: AsyncIterator[str], queue: asyncio.Queue, result: SpokenResult,
                       slots: asyncio.Semaphore) -> None:
        chunker = SentenceChunker(self.first_clause_chars)
        try:
            async for delta in tokens:
                if result.t_first_token is None:
                    result.t_first_token = time.monotonic()
                result.generated += delta
                if self._interrupt:
                    return
                for sentence in chunker.feed(delta):
                    if not await self._enqueue(sentence, queue, result, slots):
                        return
            if not self._interrupt:
                for sentence in chunker.flush():
                    if not await self._enqueue(sentence, queue, result, slots):
                        return
        except LLMRefusal as exc:
            result.filtered, result.filter_reason = True, f"refusal: {exc}"
            await self._enqueue_deflection(queue, result, slots)
        except LLMError as exc:
            result.error = str(exc)
            log.warning("%s: generation failed: %s", self.name, exc)
        finally:
            queue.put_nowait(None)
            closer = getattr(tokens, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass

    def _prepare(self, raw: str) -> tuple[str, list[str]]:
        raw = strip_speaker_prefix(raw, [self.name])
        text, tags = extract_tags(raw, self.persona.emotions)
        return clean_for_tts(text), tags

    async def _enqueue(self, raw: str, queue: asyncio.Queue, result: SpokenResult, slots: asyncio.Semaphore) -> bool:
        text, tags = self._prepare(raw)
        if not has_speakable_content(text):
            self._carry_tags.extend(tags)
            return True
        tags, self._carry_tags = self._carry_tags + tags, []
        if self.repetition.is_repetitive(text):
            result.skipped_repetitive += 1
            self.bus.publish("repetition", {"speaker": self.name, "text": text})
            return True
        verdict = self.output_filter.check(text)
        if not verdict.allowed:
            result.filtered, result.filter_reason = True, verdict.reason
            self.bus.publish("filtered", {"speaker": self.name, "reason": verdict.reason, "stage": "output"})
            await self._enqueue_deflection(queue, result, slots)
            return False
        await slots.acquire()
        moderation = asyncio.ensure_future(self.moderator.check(verdict.text)) if self.moderator else None
        queue.put_nowait(_Utterance(verdict.text, tags, asyncio.ensure_future(self._synth(verdict.text, tags)),
                                    moderation))
        result.enqueued += 1
        return result.enqueued < self.max_sentences

    def _pick_deflection(self) -> str:
        options = [d for d in self.persona.deflections if d != self._last_deflection] or self.persona.deflections
        line = self.rng.choice(options) if options else "Let's talk about something else."
        self._last_deflection = line
        return line

    async def _enqueue_deflection(self, queue: asyncio.Queue, result: SpokenResult, slots: asyncio.Semaphore) -> None:
        text, tags = self._prepare(self._pick_deflection())
        await slots.acquire()
        queue.put_nowait(_Utterance(text, tags, asyncio.ensure_future(self._synth(text, tags)), deflection=True))
        result.enqueued += 1

    async def _synth(self, text: str, tags: list[str]) -> AudioClip:
        return await asyncio.wait_for(self.tts.synthesize(text, self.emotion.voice_style(tags)), self.synth_timeout_s)

    async def _consume(self, queue: asyncio.Queue, result: SpokenResult, slots: asyncio.Semaphore) -> None:
        extra_used = False
        while True:
            item: Optional[_Utterance] = await queue.get()
            if item is None:
                return
            try:
                mode = self._interrupt
                if mode in ("now", "after_sentence"):
                    item.synth.cancel()
                    return
                if mode == "soon":
                    if extra_used:
                        item.synth.cancel()
                        return
                    extra_used = True
                if item.moderation is not None:
                    verdict: OutputVerdict = await item.moderation
                    if not verdict.allowed:
                        item.synth.cancel()
                        result.filtered, result.filter_reason = True, verdict.reason
                        self.bus.publish("filtered", {"speaker": self.name, "reason": verdict.reason, "stage": "moderation"})
                        text, tags = self._prepare(self._pick_deflection())
                        item = _Utterance(text, tags, asyncio.ensure_future(self._synth(text, tags)), deflection=True)
                        await self._play(item, result)
                        return
                if not await self._play(item, result):
                    return
                if item.deflection:
                    return
                if self._interrupt in ("now", "after_sentence"):
                    return
                await _call(self.hooks.on_speech_finished, self.name, False, False, None)
            finally:
                slots.release()

    async def _play(self, item: _Utterance, result: SpokenResult) -> bool:
        """Play one utterance. Returns False if playback was cut off or synthesis failed."""
        try:
            clip = await item.synth
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            log.warning("%s: TTS failed for %r: %s", self.name, item.text[:40], exc)
            result.error = result.error or f"tts: {exc}"
            return True  # skip this sentence, keep going
        if self._interrupt == "now":
            return False
        emotion = self.emotion.apply_tags(item.tags)
        if emotion:
            result.emotions.append(emotion)
            await _call(self.hooks.on_expression, emotion)
        if result.t_first_audio is None:
            result.t_first_audio = time.monotonic()
        self.bus.publish("caption", {"speaker": self.name, "text": item.text, "emotion": emotion or self.emotion.emotion,
                                     "deflection": item.deflection})
        await _call(self.hooks.on_sentence, self.name, item.text, item.tags, clip)
        completed = await self.player.play(clip, on_level=self.hooks.on_level)
        self.repetition.record(item.text)
        if completed:
            result.spoken.append(item.text)
            return True
        result.spoken.append(item.text.rstrip(".!?") + "—")
        result.interrupted = True
        return False
