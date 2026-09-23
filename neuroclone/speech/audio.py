"""Audio clips and players. Lip-sync levels come from the clip's RMS envelope."""

from __future__ import annotations

import abc
import asyncio
import io
import logging
import struct
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import AudioConfig

log = logging.getLogger(__name__)

LevelCallback = Callable[[float], None]


@dataclass
class AudioClip:
    samples: np.ndarray  # float32 mono, [-1, 1]
    sample_rate: int
    text: str = ""
    envelope_hint: Optional[np.ndarray] = field(default=None, repr=False)  # precomputed levels at 60 fps

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0

    @classmethod
    def silence(cls, seconds: float, sample_rate: int = 16000) -> "AudioClip":
        return cls(np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)

    @classmethod
    def from_pcm16(cls, data: bytes, sample_rate: int, channels: int = 1) -> "AudioClip":
        arr = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            arr = arr[: len(arr) - len(arr) % channels].reshape(-1, channels).mean(axis=1)
        return cls(arr, sample_rate)

    @classmethod
    def from_wav(cls, data: bytes) -> "AudioClip":
        """Parse RIFF/WAVE with PCM (8/16/24/32-bit) or IEEE float samples."""
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("not a WAV file")
        pos, fmt, raw = 12, None, None
        while pos + 8 <= len(data):
            cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
            body = data[pos + 8 : pos + 8 + size]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", body[:16])
            elif cid == b"data":
                raw = body if size != 0xFFFFFFFF else data[pos + 8 :]  # streamed WAVs may lie about size
            pos += 8 + size + (size & 1)
            if raw is not None and fmt is not None:
                break
        if fmt is None or raw is None:
            raise ValueError("WAV is missing fmt or data chunk")
        tag, channels, rate, _, _, bits = fmt
        if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: treat by bit depth
            tag = 3 if bits == 32 and len(raw) % 4 == 0 and _looks_float(raw) else 1
        if tag == 3:
            arr = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype="<f4").astype(np.float32)
        elif bits == 16:
            arr = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 8:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 24:
            usable = raw[: len(raw) - len(raw) % 3]
            b = np.frombuffer(usable, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
            arr = ints.astype(np.float32) / 8388608.0
        elif bits == 32:
            arr = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV bit depth {bits}")
        if channels > 1:
            arr = arr[: len(arr) - len(arr) % channels].reshape(-1, channels).mean(axis=1)
        return cls(np.clip(arr, -1.0, 1.0), rate)

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes((np.clip(self.samples, -1, 1) * 32767).astype("<i2").tobytes())
        return buf.getvalue()

    def resample(self, target_rate: int) -> "AudioClip":
        if target_rate == self.sample_rate or len(self.samples) == 0:
            return AudioClip(self.samples, target_rate if len(self.samples) == 0 else self.sample_rate, self.text,
                             self.envelope_hint)
        n = int(round(len(self.samples) * target_rate / self.sample_rate))
        x_old = np.linspace(0.0, 1.0, num=len(self.samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return AudioClip(np.interp(x_new, x_old, self.samples).astype(np.float32), target_rate, self.text,
                         self.envelope_hint)

    def envelope(self, fps: int = 60) -> np.ndarray:
        """Mouth-open levels in [0, 1], one per video frame, with fast attack and slow release."""
        if self.envelope_hint is not None:
            hint = self.envelope_hint
            if fps != 60 and len(hint):
                hint = np.interp(np.linspace(0, len(hint) - 1, int(len(hint) * fps / 60)), np.arange(len(hint)), hint)
            return np.clip(hint, 0, 1).astype(np.float32)
        hop = max(1, int(self.sample_rate / fps))
        frames = len(self.samples) // hop
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        rms = np.sqrt(np.mean(self.samples[: frames * hop].reshape(frames, hop) ** 2, axis=1))
        ref = float(np.percentile(rms, 95)) if np.any(rms > 0) else 0.0
        level = np.clip(rms / ref, 0.0, 1.0) if ref > 1e-5 else np.zeros_like(rms)
        level[level < 0.08] = 0.0
        out = np.zeros_like(level)
        prev = 0.0
        for i, v in enumerate(level):
            prev = v if v > prev else prev * 0.65 + v * 0.35
            out[i] = prev
        return out.astype(np.float32)


def _looks_float(raw: bytes) -> bool:
    sample = np.frombuffer(raw[: min(len(raw), 4096) - min(len(raw), 4096) % 4], dtype="<f4")
    return bool(sample.size) and bool(np.all(np.isfinite(sample))) and float(np.max(np.abs(sample))) <= 1.5


class AudioPlayer(abc.ABC):
    fps = 30

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    @abc.abstractmethod
    async def play(self, clip: AudioClip, on_level: Optional[LevelCallback] = None) -> bool:
        """Play a clip; returns False if it was stopped early."""

    def stop(self) -> None:
        self._stop.set()

    async def aclose(self) -> None:
        return None


class NullPlayer(AudioPlayer):
    """Headless player: keeps real-time pacing (scaled by ``time_scale``) and drives lip-sync."""

    def __init__(self, time_scale: float = 1.0, fps: int = 30) -> None:
        super().__init__()
        self.time_scale = time_scale
        self.fps = fps
        self.played: list[AudioClip] = []

    async def play(self, clip: AudioClip, on_level: Optional[LevelCallback] = None) -> bool:
        self._stop.clear()
        self.played.append(clip)
        if self.time_scale <= 0:
            await asyncio.sleep(0)
            if on_level:
                on_level(0.0)
            return not self._stop.is_set()
        env = clip.envelope(self.fps)
        loop = asyncio.get_running_loop()
        start = loop.time()
        total = clip.duration * self.time_scale
        completed = True
        while True:
            elapsed = loop.time() - start
            if elapsed >= total:
                break
            if self._stop.is_set():
                completed = False
                break
            if on_level is not None and len(env):
                on_level(float(env[min(len(env) - 1, int(elapsed / self.time_scale * self.fps))]))
            await asyncio.sleep(min(1.0 / self.fps, total - elapsed))
        if on_level is not None:
            on_level(0.0)
        return completed


class WavDumpPlayer(NullPlayer):
    """Writes every clip to disk (for debugging voices) while keeping real-time pacing."""

    def __init__(self, folder: str | Path, time_scale: float = 1.0) -> None:
        super().__init__(time_scale)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._n = 0

    async def play(self, clip: AudioClip, on_level: Optional[LevelCallback] = None) -> bool:
        self._n += 1
        path = self.folder / f"{time.strftime('%H%M%S')}_{self._n:04d}.wav"
        await asyncio.to_thread(path.write_bytes, clip.to_wav_bytes())
        return await super().play(clip, on_level)


class SoundDevicePlayer(AudioPlayer):
    """Real playback through PortAudio (``pip install sounddevice``)."""

    fps = 60

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()
        import sounddevice as sd  # noqa: F401 - fail fast if missing

        self.sd = sd
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device

    async def play(self, clip: AudioClip, on_level: Optional[LevelCallback] = None) -> bool:
        self._stop.clear()
        sd = self.sd
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        data = clip.samples.astype(np.float32)
        pos = 0

        def callback(outdata, frames, _time, _status):
            nonlocal pos
            chunk = data[pos : pos + frames]
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk) :, 0] = 0
            pos += frames
            if pos >= len(data):
                raise sd.CallbackStop

        stream = sd.OutputStream(
            samplerate=clip.sample_rate, channels=1, dtype="float32", callback=callback, device=self.device,
            finished_callback=lambda: loop.call_soon_threadsafe(done.set),
        )
        env = clip.envelope(self.fps)
        completed = True
        stream.start()
        start = loop.time()
        try:
            while not done.is_set():
                if self._stop.is_set():
                    completed = False
                    stream.abort()
                    break
                if on_level is not None and len(env):
                    on_level(float(env[min(len(env) - 1, int((loop.time() - start) * self.fps))]))
                await asyncio.sleep(1.0 / self.fps)
        finally:
            stream.close()
            if on_level is not None:
                on_level(0.0)
        return completed


def create_player(cfg: AudioConfig) -> AudioPlayer:
    kind = (cfg.player or "auto").lower()
    if kind == "null":
        return NullPlayer(cfg.time_scale)
    if kind == "wav":
        return WavDumpPlayer(cfg.wav_dir, cfg.time_scale)
    try:
        player = SoundDevicePlayer(cfg.device)
        player.sd.query_devices(cfg.device, kind="output")
        return player
    except Exception as exc:  # noqa: BLE001 - no library, no device, or PortAudio missing
        if kind == "sounddevice":
            raise RuntimeError(f"sounddevice playback unavailable: {exc}") from exc
        log.warning("no audio output (%s); running headless with simulated playback", exc)
        return NullPlayer(cfg.time_scale)
