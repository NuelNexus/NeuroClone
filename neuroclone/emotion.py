"""A continuous mood model (valence/arousal) that drives the avatar, the voice and the prompt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

TAG_IMPULSES: dict[str, tuple[float, float]] = {
    "happy": (0.25, 0.15),
    "excited": (0.30, 0.35),
    "smug": (0.12, 0.10),
    "love": (0.35, 0.10),
    "sad": (-0.35, -0.10),
    "angry": (-0.30, 0.30),
    "scared": (-0.30, 0.30),
    "confused": (-0.05, 0.10),
    "thinking": (0.0, -0.05),
    "surprised": (0.05, 0.30),
    "tired": (-0.10, -0.30),
    "neutral": (0.0, 0.0),
}

EVENT_IMPULSES: dict[str, tuple[float, float]] = {
    "sub": (0.30, 0.30),
    "resub": (0.25, 0.20),
    "gift": (0.40, 0.40),
    "raid": (0.30, 0.50),
    "bits": (0.25, 0.25),
    "superchat": (0.30, 0.30),
    "member": (0.30, 0.30),
    "donation": (0.30, 0.30),
    "follow": (0.10, 0.10),
    "insult": (-0.15, 0.10),
    "game_win": (0.40, 0.40),
    "game_loss": (-0.30, 0.20),
    "filtered": (-0.05, 0.05),
}

# How each emotion colours the voice: (rate %, pitch %).
VOICE_SHIFTS: dict[str, tuple[float, float]] = {
    "excited": (7, 4),
    "happy": (3, 2),
    "smug": (-2, -1),
    "love": (-2, 2),
    "sad": (-7, -4),
    "angry": (4, -2),
    "scared": (6, 5),
    "confused": (-2, 2),
    "thinking": (-4, -1),
    "surprised": (5, 6),
    "tired": (-8, -3),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class VoiceStyle:
    rate_pct: float = 0.0
    pitch_pct: float = 0.0
    emotion: str = "neutral"


@dataclass
class Mood:
    valence: float
    arousal: float
    label: str
    emotion: str

    def describe(self) -> str:
        energy = "high" if self.arousal > 0.6 else "low" if self.arousal < 0.3 else "medium"
        return f"{self.label} (energy {energy})"


class EmotionEngine:
    def __init__(self, baseline_valence: float = 0.2, baseline_arousal: float = 0.45, half_life_s: float = 90.0) -> None:
        self.base_v = baseline_valence
        self.base_a = baseline_arousal
        self.half_life_s = half_life_s
        self.valence = baseline_valence
        self.arousal = baseline_arousal
        self.emotion = "neutral"
        self._t = time.time()

    def _decay(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        dt = max(0.0, now - self._t)
        self._t = now
        if dt <= 0:
            return
        factor = 0.5 ** (dt / self.half_life_s)
        self.valence = self.base_v + (self.valence - self.base_v) * factor
        self.arousal = self.base_a + (self.arousal - self.base_a) * factor
        if factor < 0.25:
            self.emotion = "neutral"

    def _push(self, dv: float, da: float, strength: float, now: Optional[float]) -> None:
        self._decay(now)
        self.valence = _clamp(self.valence + dv * strength, -1.0, 1.0)
        self.arousal = _clamp(self.arousal + da * strength, 0.0, 1.0)

    def apply_tags(self, tags: Iterable[str], now: Optional[float] = None) -> Optional[str]:
        last = None
        for tag in tags:
            if tag in TAG_IMPULSES:
                self._push(*TAG_IMPULSES[tag], 1.0, now)
                self.emotion = last = tag
        return last

    def apply_event(self, kind: str, magnitude: float = 1.0, now: Optional[float] = None) -> None:
        if kind in EVENT_IMPULSES:
            self._push(*EVENT_IMPULSES[kind], _clamp(magnitude, 0.2, 2.0), now)

    def mood(self, now: Optional[float] = None) -> Mood:
        self._decay(now)
        v, a = self.valence, self.arousal
        if v > 0.35 and a > 0.6:
            label = "hyped"
        elif v > 0.35:
            label = "cheerful"
        elif v < -0.3 and a > 0.5:
            label = "fired up"
        elif v < -0.3:
            label = "gloomy"
        elif a < 0.25:
            label = "sleepy"
        elif a > 0.65:
            label = "restless"
        elif v > 0.1:
            label = "in a good mood"
        else:
            label = "chill"
        return Mood(v, a, label, self.emotion)

    def smile(self) -> float:
        return _clamp((self.valence + 1) / 2, 0.0, 1.0)

    def voice_style(self, tags: Iterable[str] = ()) -> VoiceStyle:
        self._decay()
        emotion = next((t for t in reversed(list(tags)) if t in VOICE_SHIFTS), self.emotion)
        rate, pitch = VOICE_SHIFTS.get(emotion, (0.0, 0.0))
        rate += (self.arousal - self.base_a) * 10
        return VoiceStyle(rate_pct=round(_clamp(rate, -15, 15), 1), pitch_pct=round(_clamp(pitch, -10, 10), 1),
                          emotion=emotion)
