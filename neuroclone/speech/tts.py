"""Text-to-speech backends. Each returns an AudioClip for one sentence.

Voices come from the persona card (``voice:`` per provider) and can be overridden in config.
Emotion shifts rate/pitch per sentence through ``VoiceStyle``.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import re
from typing import Optional
from xml.sax.saxutils import escape

import aiohttp
import numpy as np

from ..config import TTSConfig
from ..emotion import VoiceStyle
from ..persona import Persona
from .audio import AudioClip

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


def _parse_signed(value: str, unit: str) -> float:
    m = re.fullmatch(rf"\s*([+-]?\d+(?:\.\d+)?)\s*{re.escape(unit)}\s*", value or "")
    return float(m.group(1)) if m else 0.0


def _fmt_signed(value: float, unit: str) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(round(value)):.0f}{unit}"


class TTS(abc.ABC):
    name = "tts"

    @abc.abstractmethod
    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        ...

    async def aclose(self) -> None:
        return None


class SilentTTS(TTS):
    """No audio, realistic timing, and a syllable-shaped envelope so lip-sync still animates."""

    name = "silent"

    def __init__(self, chars_per_second: float = 14.0, sample_rate: int = 16000) -> None:
        self.cps = chars_per_second
        self.sample_rate = sample_rate

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        rate = 1.0 + (style.rate_pct / 100 if style else 0.0)
        seconds = max(0.4, len(text) / (self.cps * rate) + 0.25)
        frames = int(seconds * 60)
        t = np.arange(frames) / 60.0
        env = np.clip(0.55 + 0.45 * np.sin(2 * np.pi * 4.2 * t) * np.sin(2 * np.pi * 0.9 * t + 1), 0, 1)
        env[-6:] = 0
        clip = AudioClip.silence(seconds, self.sample_rate)
        clip.text = text
        clip.envelope_hint = env.astype(np.float32)
        return clip


class _HTTPTTS(TTS):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(trust_env=True)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class AzureTTS(_HTTPTTS):
    """Azure Neural TTS over REST with SSML prosody (the technique behind Neuro-sama's voice)."""

    name = "azure"

    def __init__(self, key: str, region: str, voice: str, pitch: str = "+0%", rate: str = "+0%",
                 timeout_s: float = 20.0) -> None:
        super().__init__(timeout_s)
        if not key:
            raise TTSError("Azure TTS needs tts.azure_key (or AZURE_SPEECH_KEY)")
        self.key, self.region, self.voice = key, region, voice or "en-US-SaraNeural"
        self.base_pitch = _parse_signed(pitch, "%")
        self.base_rate = _parse_signed(rate, "%")

    def ssml(self, text: str, style: Optional[VoiceStyle] = None) -> str:
        pitch = self.base_pitch + (style.pitch_pct if style else 0)
        rate = self.base_rate + (style.rate_pct if style else 0)
        lang = "-".join(self.voice.split("-")[:2])
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang}">'
            f'<voice name="{escape(self.voice)}"><prosody pitch="{_fmt_signed(pitch, "%")}" '
            f'rate="{_fmt_signed(rate, "%")}">{escape(text)}</prosody></voice></speak>'
        )

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        session = await self._session_get()
        url = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
            "User-Agent": "neuroclone",
        }
        try:
            async with session.post(url, data=self.ssml(text, style).encode("utf-8"), headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as resp:
                body = await resp.read()
                if resp.status != 200:
                    raise TTSError(f"Azure TTS returned {resp.status}: {body[:200]!r}")
        except aiohttp.ClientError as exc:
            raise TTSError(f"Azure TTS unreachable: {exc}") from exc
        clip = AudioClip.from_wav(body)
        clip.text = text
        return clip


class OpenAITTS(_HTTPTTS):
    """OpenAI-compatible /audio/speech: Kokoro-FastAPI, OpenAI, and many local servers."""

    name = "openai"

    def __init__(self, base_url: str, model: str, voice: str, api_key: str = "", speed: float = 1.0,
                 timeout_s: float = 20.0, pitch_semitones: float = 0.0) -> None:
        super().__init__(timeout_s)
        self.base_url, self.model, self.voice, self.api_key, self.speed = (
            base_url.rstrip("/"), model, voice or "af_bella", api_key, speed)
        self.pitch_semitones = pitch_semitones

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        session = await self._session_get()
        speed = self.speed * (1 + (style.rate_pct / 100 if style else 0))
        ratio = pitch_ratio(self.pitch_semitones, style) if self.pitch_semitones else 1.0
        body = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav",
                "speed": round(speed_and_pitch(speed, ratio), 3)}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with session.post(f"{self.base_url}/audio/speech", json=body, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as resp:
                data = await resp.read()
                if resp.status != 200:
                    raise TTSError(f"TTS server returned {resp.status}: {data[:200]!r}")
        except aiohttp.ClientError as exc:
            raise TTSError(f"TTS server {self.base_url} unreachable: {exc}") from exc
        clip = AudioClip.from_wav(data) if data[:4] == b"RIFF" else AudioClip.from_pcm16(data, 24000)
        if ratio != 1.0:
            clip.samples = apply_ratio(clip.samples, ratio)
        clip.text = text
        return clip


class EdgeTTS(TTS):
    """Free Microsoft Edge online voices (``pip install 'neuroclone[edge-tts]'``)."""

    name = "edge"

    def __init__(self, voice: str, pitch: str = "+0Hz", rate: str = "+0%") -> None:
        try:
            import edge_tts  # noqa: F401
            import miniaudio  # noqa: F401
        except ImportError as exc:
            raise TTSError("edge TTS needs: pip install 'neuroclone[edge-tts]'") from exc
        self.voice = voice or "en-US-EmmaNeural"
        self.base_pitch = _parse_signed(pitch, "Hz")
        self.base_rate = _parse_signed(rate, "%")

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        import edge_tts
        import miniaudio

        pitch = self.base_pitch + (style.pitch_pct * 2 if style else 0)  # ~2 Hz per % for a ~200 Hz voice
        rate = self.base_rate + (style.rate_pct if style else 0)
        comm = edge_tts.Communicate(text, self.voice, rate=_fmt_signed(rate, "%"), pitch=_fmt_signed(pitch, "Hz"))
        mp3 = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            raise TTSError("edge-tts returned no audio")
        decoded = await asyncio.to_thread(
            miniaudio.decode, bytes(mp3), output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=24000
        )
        return AudioClip(np.asarray(decoded.samples, dtype=np.float32), decoded.sample_rate, text)


# Kokoro voice-name prefix -> espeak language (af_bella = American English female, bf_ = British...).
KOKORO_LANGS = {"a": "en-us", "b": "en-gb", "j": "ja", "z": "cmn", "e": "es", "f": "fr-fr", "h": "hi", "i": "it",
                "p": "pt-br"}
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"


def pitch_ratio(semitones: float, style: Optional[VoiceStyle] = None) -> float:
    """Frequency ratio for a base shift in semitones plus the emotion's percentage."""
    ratio = 2.0 ** (semitones / 12.0)
    if style is not None and style.pitch_pct:
        ratio *= 1.0 + style.pitch_pct / 100.0
    return ratio


def resample_to(samples: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited (FFT) resampling to exactly ``n_out`` samples."""
    n_in = len(samples)
    if n_out == n_in or n_in == 0 or n_out <= 0:
        return samples.astype(np.float32)
    spectrum = np.fft.rfft(samples)
    out = np.zeros(n_out // 2 + 1, dtype=spectrum.dtype)
    keep = min(len(spectrum), len(out))
    out[:keep] = spectrum[:keep]
    return (np.fft.irfft(out, n_out) * (n_out / n_in)).astype(np.float32)


def speed_and_pitch(speed: float, ratio: float) -> float:
    """The 'tape' trick: ask the voice model to speak ``ratio`` times slower, then resample the
    result ``ratio`` times shorter. Duration comes back to normal and pitch rises by ``ratio``, with
    none of the warble of a time-stretching pitch shifter (the neural model did the stretching)."""
    return max(0.5, min(2.0, speed / ratio))


def apply_ratio(samples: np.ndarray, ratio: float) -> np.ndarray:
    if abs(ratio - 1.0) < 1e-3:
        return samples.astype(np.float32)
    return resample_to(samples, max(1, int(round(len(samples) / ratio))))


def parse_voice_mix(spec: str) -> list[tuple[str, float]]:
    """``"af_bella"`` or ``"af_bella:0.7+af_sky:0.3"`` -> [(name, weight)], weights normalised."""
    parts = []
    for chunk in (spec or "").split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, weight = chunk.partition(":")
        try:
            parts.append((name.strip(), float(weight) if weight else 1.0))
        except ValueError as exc:
            raise TTSError(f"bad voice weight in {spec!r}") from exc
    total = sum(w for _, w in parts)
    if not parts or total <= 0:
        raise TTSError(f"no voice in {spec!r}")
    return [(n, w / total) for n, w in parts]


def onnx_providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if (device or "cpu").lower() in ("gpu", "cuda", "auto"):
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"]
        chosen = [p for p in preferred if p in available]
        if chosen:
            return chosen + ["CPUExecutionProvider"]
        log.info("no GPU build of onnxruntime installed; the voice runs on the CPU")
    return ["CPUExecutionProvider"]


class KokoroTTS(TTS):
    """Kokoro-82M in-process via ONNX Runtime (``pip install 'neuroclone[kokoro]'``): offline, free
    (Apache-2.0), 24 kHz, faster than real time on a desktop CPU so the GPU stays free for the LLM.
    Voices can be blended (``af_bella:0.7+af_sky:0.3``) and pitched (``pitch_semitones``)."""

    name = "kokoro"

    def __init__(self, voice: str, *, model_path: str, voices_path: str, lang: str = "", speed: float = 1.0,
                 pitch_semitones: float = 0.0, device: str = "cpu", threads: int = 0) -> None:
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise TTSError("the kokoro voice needs: pip install 'neuroclone[kokoro]'") from exc
        from pathlib import Path

        missing = [p for p in (model_path, voices_path) if not Path(p).exists()]
        if missing:
            raise TTSError(f"Kokoro files not found: {', '.join(missing)}. Run `neuroclone setup` or download "
                           f"kokoro-v1.0.onnx and voices-v1.0.bin from {KOKORO_URL}")
        options = ort.SessionOptions()
        options.log_severity_level = 3  # the fp16 model is chatty about constant folding
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(model_path, sess_options=options, providers=onnx_providers(device))
        self.kokoro = Kokoro.from_session(session, voices_path)
        self.voice_spec = voice or "af_bella"
        self.style = self._resolve(self.voice_spec)
        first = parse_voice_mix(self.voice_spec)[0][0]
        self.lang = KOKORO_LANGS.get(lang, lang) if lang else KOKORO_LANGS.get(first[:1], "en-us")
        self.speed = speed
        self.pitch_semitones = pitch_semitones
        self._lock = asyncio.Lock()

    def _resolve(self, spec: str):
        mix = parse_voice_mix(spec)
        known = set(self.kokoro.get_voices())
        unknown = [n for n, _ in mix if n not in known]
        if unknown:
            raise TTSError(f"unknown Kokoro voice {', '.join(unknown)}; try one of: {', '.join(sorted(known)[:30])}")
        if len(mix) == 1:
            return mix[0][0]
        return sum(w * self.kokoro.get_voice_style(n) for n, w in mix)

    def _run(self, text: str, speed: float) -> np.ndarray:
        samples, _ = self.kokoro.create(text, voice=self.style, speed=speed, lang=self.lang)
        return np.asarray(samples, dtype=np.float32)

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        speed = self.speed * (1 + (style.rate_pct / 100 if style else 0))
        ratio = pitch_ratio(self.pitch_semitones, style)
        async with self._lock:  # one inference at a time: parallel runs only fight over the same cores
            samples = await asyncio.to_thread(self._run, text, speed_and_pitch(speed, ratio))
        return AudioClip(apply_ratio(samples, ratio), 24000, text)

    async def warmup(self) -> None:
        await self.synthesize("Hi.")


class KokoroTorchTTS(TTS):
    """Kokoro through the original PyTorch package (``pip install kokoro``), if that's what you have."""

    name = "kokoro"

    def __init__(self, voice: str, lang_code: str = "a", speed: float = 1.0, pitch_semitones: float = 0.0) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSError("the kokoro voice needs: pip install 'neuroclone[kokoro]'") from exc
        self.pipeline = KPipeline(lang_code=lang_code or (voice or "a")[:1])
        self.voice = voice or "af_bella"
        self.speed = speed
        self.pitch_semitones = pitch_semitones
        self._lock = asyncio.Lock()

    def _run(self, text: str, speed: float) -> np.ndarray:
        parts = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=speed):
            parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    async def synthesize(self, text: str, style: Optional[VoiceStyle] = None) -> AudioClip:
        speed = self.speed * (1 + (style.rate_pct / 100 if style else 0))
        ratio = pitch_ratio(self.pitch_semitones, style)
        async with self._lock:  # the pipeline is not re-entrant
            samples = await asyncio.to_thread(self._run, text, speed_and_pitch(speed, ratio))
        return AudioClip(apply_ratio(samples, ratio), 24000, text)


def create_tts(cfg: TTSConfig, persona: Persona) -> TTS:
    provider = (cfg.provider or "silent").lower()
    voice_cfg = persona.voice_for(provider)
    voice = cfg.voice or voice_cfg.get("voice", "")
    pitch = cfg.pitch or voice_cfg.get("pitch", "")
    rate = cfg.rate or voice_cfg.get("rate", "")
    speed = float(voice_cfg.get("speed", 1.0))
    semitones = cfg.pitch_semitones + float(voice_cfg.get("pitch_semitones", 0.0))
    if provider == "silent":
        return SilentTTS(cfg.chars_per_second)
    if provider == "azure":
        return AzureTTS(cfg.azure_key, cfg.azure_region, voice, pitch or "+0%", rate or "+0%", cfg.timeout_s)
    if provider == "openai":
        return OpenAITTS(cfg.base_url, cfg.model, voice, cfg.api_key, speed, cfg.timeout_s, pitch_semitones=semitones)
    if provider == "edge":
        return EdgeTTS(voice, pitch or "+0Hz", rate or "+0%")
    if provider == "kokoro":
        import importlib.util

        if importlib.util.find_spec("kokoro_onnx") is None and importlib.util.find_spec("kokoro") is not None:
            return KokoroTorchTTS(voice, cfg.kokoro_lang, speed, semitones)
        return KokoroTTS(voice, model_path=cfg.kokoro_model, voices_path=cfg.kokoro_voices, lang=cfg.kokoro_lang,
                         speed=speed, pitch_semitones=semitones, device=cfg.device, threads=cfg.threads)
    raise TTSError(f"unknown tts.provider {cfg.provider!r} (silent, kokoro, openai, azure, edge)")
