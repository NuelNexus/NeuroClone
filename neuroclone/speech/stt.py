"""Speech recognition: a faster-whisper transcriber and a microphone listener with barge-in.

``pip install 'neuroclone[stt]'``. The listener uses an adaptive energy VAD, emits
``SpeechStarted`` as soon as someone starts talking (so the character can stop and listen),
and emits ``VoiceTranscript`` when the utterance ends.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from collections import deque
from typing import Callable, Optional

import numpy as np

from ..config import STTConfig
from ..events import SpeechStarted, VoiceTranscript
from ..repetition import _grams, jaccard

log = logging.getLogger(__name__)

# Whisper tends to invent these on noise or silence.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "you", "bye.", "bye",
    "subtitles by the amara.org community", ".", "okay.", "so",
}


class STTError(RuntimeError):
    pass


def resample(samples: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(samples) == 0:
        return samples.astype(np.float32)
    n = int(round(len(samples) * dst / src))
    return np.interp(np.linspace(0, len(samples), n, endpoint=False), np.arange(len(samples)), samples).astype(np.float32)


class Transcriber(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        ...


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model: str = "base.en", device: str = "auto", compute_type: str = "default",
                 language: str = "en") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError("speech recognition needs: pip install 'neuroclone[stt]'") from exc
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language or None
        self._lock = asyncio.Lock()

    def _run(self, samples: np.ndarray) -> str:
        segments, _ = self.model.transcribe(samples, language=self.language, beam_size=1,
                                            condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        audio = resample(samples, sample_rate, 16000)
        if len(audio) < 1600:
            return ""
        async with self._lock:
            text = await asyncio.to_thread(self._run, audio)
        return "" if text.lower().strip() in _HALLUCINATIONS else text


class MicrophoneListener:
    BLOCK_MS = 30
    RATE = 16000

    def __init__(
        self,
        cfg: STTConfig,
        transcriber: Transcriber,
        submit: Callable[[object], None],
        speaker: str,
        ai_speaking: Callable[[], bool] = lambda: False,
        recent_ai_text: Callable[[], str] = lambda: "",
    ) -> None:
        self.cfg = cfg
        self.transcriber = transcriber
        self.submit = submit
        self.speaker = speaker
        self.ai_speaking = ai_speaking
        self.recent_ai_text = recent_ai_text
        self._closing = False

    def _is_echo(self, text: str) -> bool:
        """With speakers instead of headphones the mic hears the character; ignore her own words."""
        said = self.recent_ai_text()
        return bool(said) and jaccard(_grams(text), _grams(said)) > 0.35

    async def run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise STTError("microphone input needs: pip install 'neuroclone[stt]'") from exc
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=400)
        block = int(self.RATE * self.BLOCK_MS / 1000)

        def callback(indata, _frames, _time, _status):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())
            except asyncio.QueueFull:
                pass

        device = int(self.cfg.input_device) if (self.cfg.input_device or "").isdigit() else self.cfg.input_device
        noise = self.cfg.vad_threshold / 3
        pre_roll: deque = deque(maxlen=10)
        buf: list[np.ndarray] = []
        speaking, announced = False, False
        speech_ms = silence_ms = 0
        with sd.InputStream(samplerate=self.RATE, channels=1, dtype="float32", blocksize=block, device=device,
                            callback=callback):
            log.info("microphone listening as %s", self.speaker)
            while not self._closing:
                chunk = await queue.get()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                threshold = max(self.cfg.vad_threshold, noise * 3.0)
                if self.ai_speaking():
                    threshold *= 2.5  # only a clearly louder voice counts as barge-in
                if rms > threshold:
                    if not speaking:
                        speaking, announced, speech_ms, buf = True, False, 0, list(pre_roll)
                    buf.append(chunk)
                    speech_ms += self.BLOCK_MS
                    silence_ms = 0
                    if not announced and speech_ms >= self.cfg.min_speech_ms:
                        announced = True
                        if self.cfg.barge_in:
                            self.submit(SpeechStarted(self.speaker))
                elif speaking:
                    buf.append(chunk)
                    silence_ms += self.BLOCK_MS
                    if silence_ms >= self.cfg.silence_ms:
                        speaking = False
                        if speech_ms >= self.cfg.min_speech_ms:
                            audio = np.concatenate(buf)
                            try:
                                text = await self.transcriber.transcribe(audio, self.RATE)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("transcription failed: %s", exc)
                                text = ""
                            if text and not self._is_echo(text):
                                self.submit(VoiceTranscript(self.speaker, text, source="mic"))
                        buf = []
                else:
                    noise = 0.95 * noise + 0.05 * rms
                    pre_roll.append(chunk)

    async def aclose(self) -> None:
        self._closing = True


def create_transcriber(cfg: STTConfig) -> Optional[Transcriber]:
    if not cfg.enabled:
        return None
    try:
        return FasterWhisperTranscriber(cfg.model, cfg.device, cfg.compute_type, cfg.language)
    except Exception as exc:  # noqa: BLE001
        log.error("speech recognition disabled: %s", exc)
        return None
