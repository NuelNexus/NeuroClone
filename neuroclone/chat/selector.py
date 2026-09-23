"""Which chat message should the character answer next?

Neuro-sama's selection logic is not public. This scorer is explicit and inspectable (the
dashboard shows every score and why). Mentions, questions, support, newcomers, novelty and
topic relevance raise a score. Spam, copypasta, links, walls of text and users who were just
answered lower it. Softmax sampling keeps her unpredictable, and "vibe" detection notices
when chat converges on one thing.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..config import SelectorConfig
from ..events import ChatMessage
from ..memory.embeddings import HashingEmbedder
from ..safety.filter import normalize

_QUESTION = re.compile(r"\?|^(who|what|when|where|why|how|do|does|did|can|could|would|will|are|is|should)\b", re.I)
_EMOTE_ONLY = re.compile(r"^[\W_]*$|^(?:[A-Z][a-z]+[A-Z]\w*\s*)+$")  # symbols, or CamelCase emote spam


@dataclass
class Candidate:
    msg: ChatMessage
    score: float
    reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"user": self.msg.user, "text": self.msg.text, "score": round(self.score, 2),
                "reasons": {k: round(v, 2) for k, v in self.reasons.items()}}


class ChatSelector:
    def __init__(self, cfg: SelectorConfig, names: Iterable[str], rng: Optional[random.Random] = None) -> None:
        self.cfg = cfg
        self.names = [n.lower() for n in names if n]
        self._name_re = re.compile(r"\b(" + "|".join(map(re.escape, self.names)) + r")\b", re.I) if self.names else None
        self.rng = rng or random.Random(cfg.seed)
        self.embedder = HashingEmbedder(256)
        self.buffer: deque[ChatMessage] = deque(maxlen=400)
        self.answered_users: dict[str, float] = {}
        self.recent_topics: deque[np.ndarray] = deque(maxlen=12)
        self.topic_vec: Optional[np.ndarray] = None
        self._last_vibe = 0.0
        self.last_scores: list[Candidate] = []

    # ------------------------------------------------------------------ input
    def add(self, msg: ChatMessage) -> None:
        self.buffer.append(msg)

    def set_topic(self, text: str) -> None:
        self.topic_vec = self.embedder.vector(text) if text else None

    def _prune(self, now: float) -> None:
        while self.buffer and now - self.buffer[0].ts > self.cfg.max_age_s:
            self.buffer.popleft()

    def pending(self, now: Optional[float] = None) -> int:
        self._prune(now or time.time())
        return len(self.buffer)

    # ------------------------------------------------------------------ scoring
    def score(self, msg: ChatMessage, now: float, dupes: Optional[Counter] = None) -> Candidate:
        reasons: dict[str, float] = {"base": 1.0}
        text = msg.text.strip()
        n = len(text)
        if self._name_re and self._name_re.search(text):
            reasons["mention"] = 3.0
        if _QUESTION.search(text):
            reasons["question"] = 1.2
        if msg.bits:
            reasons["support"] = min(4.0, 1.5 + math.log10(1 + msg.bits))
        if msg.first_time:
            reasons["newcomer"] = 1.5
        if msg.is_sub or msg.is_mod:
            reasons["badge"] = 0.3
        vec = self.embedder.vector(text)
        if self.recent_topics:
            overlap = max(float(vec @ t) for t in self.recent_topics)
            reasons["novelty"] = 1.2 * (1.0 - max(0.0, overlap))
        else:
            reasons["novelty"] = 1.2
        if self.topic_vec is not None:
            reasons["on_topic"] = 0.8 * max(0.0, float(vec @ self.topic_vec))
        if n < 4:
            reasons["too_short"] = -1.5
        elif n < 10:
            reasons["short"] = -0.5
        elif n > 200:
            reasons["wall_of_text"] = -0.8
        else:
            reasons["good_length"] = 0.3
        letters = [c for c in text if c.isalpha()]
        if n > 8 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            reasons["all_caps"] = -0.5
        if _EMOTE_ONLY.match(text):
            reasons["emote_only"] = -2.0
        if dupes is not None:
            count = dupes.get(normalize(text), 0)
            if count >= 3:
                reasons["spam"] = -2.0 if n < 40 else -3.0  # long repeated text = copypasta
        last = self.answered_users.get(msg.user.lower())
        if last is not None and now - last < self.cfg.user_cooldown_s:
            reasons["recently_answered"] = -2.5 * (1.0 - (now - last) / self.cfg.user_cooldown_s)
        raw = sum(reasons.values())
        age = max(0.0, now - msg.ts)
        fresh = 0.5 ** (age / max(self.cfg.freshness_half_life_s, 1e-3))
        final = raw * fresh if raw > 0 else raw
        reasons["freshness_x"] = fresh
        return Candidate(msg, final, reasons)

    def pick(self, now: Optional[float] = None) -> Optional[Candidate]:
        now = now or time.time()
        self._prune(now)
        if not self.buffer:
            self.last_scores = []
            return None
        dupes = Counter(normalize(m.text) for m in self.buffer)
        cands = sorted((self.score(m, now, dupes) for m in self.buffer), key=lambda c: c.score, reverse=True)
        self.last_scores = cands[:10]
        viable = [c for c in cands if c.score > 0]
        if not viable:
            return None
        if self.cfg.temperature <= 0 or len(viable) == 1:
            chosen = viable[0]
        else:
            top = viable[:8]
            m = max(c.score for c in top)
            weights = [math.exp((c.score - m) / self.cfg.temperature) for c in top]
            chosen = self.rng.choices(top, weights=weights, k=1)[0]
        self.mark_answered(chosen.msg, now)
        return chosen

    def mark_answered(self, msg: ChatMessage, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.answered_users[msg.user.lower()] = now
        self.recent_topics.append(self.embedder.vector(msg.text))
        try:
            self.buffer.remove(msg)
        except ValueError:
            pass
        # Drop near-duplicates (same question, other wording) so it isn't answered twice.
        answered = self.embedder.vector(self._without_names(msg.text))
        for other in list(self.buffer):
            same = normalize(other.text) == normalize(msg.text)
            if same or float(answered @ self.embedder.vector(self._without_names(other.text))) >= 0.8:
                self.buffer.remove(other)

    def _without_names(self, text: str) -> str:
        return self._name_re.sub(" ", text) if self._name_re else text

    # ------------------------------------------------------------------ reading the room
    def vibe(self, now: Optional[float] = None, cooldown_s: float = 45.0) -> Optional[str]:
        """Describe what chat is collectively doing, if it is converging on something."""
        now = now or time.time()
        if now - self._last_vibe < cooldown_s:
            return None
        window = [m for m in self.buffer if now - m.ts <= self.cfg.vibe_window_s]
        if len(window) < self.cfg.vibe_min_messages:
            return None
        counts = Counter(normalize(m.text) for m in window if normalize(m.text))
        if counts:
            text, count = counts.most_common(1)[0]
            if count >= max(4, int(0.4 * len(window))):
                self._last_vibe = now
                users = len({m.user for m in window if normalize(m.text) == text})
                return f'{users} people in chat are spamming "{text[:40]}" right now'
        words = Counter(w for m in window for w in set(normalize(m.text).split()) if len(w) > 3)
        if words:
            word, count = words.most_common(1)[0]
            if count >= max(4, int(0.5 * len(window))):
                self._last_vibe = now
                return f'chat keeps talking about "{word}" ({count} of the last {len(window)} messages)'
        return None

    def snapshot(self) -> list[dict]:
        return [c.as_dict() for c in self.last_scores]
