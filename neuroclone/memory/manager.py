"""MemoryManager: working memory + long-term episodic/semantic memory + viewer profiles.

Retrieval follows the Generative Agents recipe: score = relevance + recency + importance,
then MMR re-ranking for diversity. Everything slow (summaries, reflections, fact extraction)
runs as background tasks so the character never waits on memory to speak.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import MemoryConfig
from ..prompts import (
    DIARY_SYSTEM,
    FACTS_SCHEMA,
    FACTS_SYSTEM,
    REFLECT_SCHEMA,
    REFLECT_SYSTEM,
    SUMMARIZE_SYSTEM,
)
from .embeddings import Embedder
from .store import MemoryRecord, MemoryStore, UserProfile

log = logging.getLogger(__name__)

_DISCLOSURE = re.compile(
    r"\b(my name is|i'?m called|call me|i'?m from|i am from|i live in|my (cat|dog|pet|bird|fish|favou?rite|birthday|"
    r"job|wife|husband|partner|girlfriend|boyfriend|kid|son|daughter|mom|dad|brother|sister)|i have (a|an|two|three) |"
    r"i work (as|at)|i'?m a |i am a |i just (got|finished|started|graduated)|today is my|i'?m learning|i play )",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """One line in the stream's working memory, from any speaker."""

    speaker: str
    text: str
    kind: str = "chat"  # chat | voice | event | twin | character | game | director | idle | vision
    character: bool = False
    meta: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Recall:
    record: MemoryRecord
    score: float
    similarity: float


def humanize_age(seconds: float) -> str:
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} min ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} h ago"
    days = hours / 24
    if days < 14:
        return f"{int(days)} day{'s' if int(days) != 1 else ''} ago"
    weeks = days / 7
    if weeks < 9:
        return f"{int(weeks)} weeks ago"
    return f"{int(days / 30)} months ago"


def estimate_importance(text: str, kind: str = "episode", *, support: float = 0, creator: bool = False,
                        first_time: bool = False) -> float:
    score = 3.0
    if support:
        score += min(4.0, 1.5 + math.log10(1 + support))
    if creator:
        score += 2.0
    if first_time:
        score += 1.0
    if _DISCLOSURE.search(text):
        score += 2.0
    if len(text) > 120:
        score += 0.5
    if kind in ("reflection", "summary"):
        score += 2.0
    return float(max(1.0, min(10.0, score)))


class MemoryManager:
    def __init__(
        self,
        cfg: MemoryConfig,
        store: MemoryStore,
        embedder: Embedder,
        llm=None,
        *,
        character: str = "Nexa",
        session_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.character = character
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.history: list[Turn] = []
        self.summary = ""
        self._since_reflect = 0
        self._bg: set[asyncio.Task] = set()
        self._summarizing = False
        self.started = time.time()
        self.store.start_session(self.session_id, self.started)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.store.needs_reembed(self.embedder.signature):
            rows = self.store.rows_for_reembed()
            log.info("re-embedding %d memories for %s", len(rows), self.embedder.signature)
            for i in range(0, len(rows), 64):
                chunk = rows[i : i + 64]
                vecs = await self.embedder.embed([t for _, t in chunk])
                self.store.set_embeddings([mid for mid, _ in chunk], vecs, self.embedder.signature)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg_done)
        return task

    def _bg_done(self, task: asyncio.Task) -> None:
        self._bg.discard(task)
        if not task.cancelled() and task.exception():
            log.warning("memory background task failed: %r", task.exception())

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for background memory work (used at shutdown and in tests)."""
        if self._bg:
            await asyncio.wait(set(self._bg), timeout=timeout)

    # ------------------------------------------------------------------ working memory
    def add_turn(self, turn: Turn) -> Turn:
        self.history.append(turn)
        if len(self.history) > self.cfg.summarize_after and not self._summarizing:
            self._summarizing = True
            self._spawn(self._summarize_overflow())
        return turn

    def recent_turns(self, limit: Optional[int] = None) -> list[Turn]:
        return self.history[-(limit or self.cfg.history_turns):]

    async def _summarize_overflow(self) -> None:
        try:
            keep = self.cfg.history_turns
            old, self.history = self.history[:-keep], self.history[-keep:]
            if not old:
                return
            transcript = "\n".join(f"{t.speaker}: {t.text}" for t in old)
            if self.llm is None:
                self.summary = (self.summary + " " + transcript[-600:]).strip()[-1200:]
                return
            prompt = (f"Previous summary: {self.summary or '(none)'}\n\nNew conversation:\n{transcript}\n\n"
                      "Write the updated summary.")
            text = await self.llm.complete(
                SUMMARIZE_SYSTEM.format(name=self.character), [{"role": "user", "content": prompt}], max_tokens=400
            )
            self.summary = text.strip()[:1500]
            await self.remember(f"Earlier on stream: {self.summary}", kind="summary", importance=5)
        finally:
            self._summarizing = False

    # ------------------------------------------------------------------ long-term memory
    async def remember(self, text: str, *, kind: str = "episode", importance: Optional[float] = None,
                       speaker: str = "", subject: str = "", meta: Optional[dict] = None) -> int:
        vec = (await self.embedder.embed([text]))[0]
        mid = self.store.add(
            kind, text, vec,
            importance=importance if importance is not None else estimate_importance(text, kind),
            speaker=speaker, subject=subject, session=self.session_id, meta=meta,
            signature=self.embedder.signature,
        )
        if kind == "episode":
            self._since_reflect += 1
            if self.cfg.reflect_every and self._since_reflect >= self.cfg.reflect_every:
                self._since_reflect = 0
                self._spawn(self.reflect())
        return mid

    async def recall(self, query: str, *, k: Optional[int] = None, now: Optional[float] = None,
                     exclude_recent_s: float = 120.0) -> list[Recall]:
        if not query.strip() or self.store.count() == 0:
            return []
        k = k or self.cfg.recall_k
        now = now or time.time()
        qvec = (await self.embedder.embed([query]))[0]
        candidates = self.store.search(qvec, k=max(24, k * 5))
        scored: list[Recall] = []
        for rec, sim in candidates:
            if sim < self.cfg.min_similarity or now - rec.created < exclude_recent_s:
                continue
            hours = max(0.0, now - max(rec.created, rec.last_access)) / 3600
            recency = 0.5 ** (hours / max(self.cfg.recency_half_life_h, 1e-3))
            score = (self.cfg.w_relevance * sim + self.cfg.w_recency * recency
                     + self.cfg.w_importance * rec.importance / 10)
            scored.append(Recall(rec, score, sim))
        chosen = self._mmr(scored, k)
        self.store.touch([r.record.id for r in chosen], now)
        return chosen

    def _mmr(self, items: list[Recall], k: int, lam: float = 0.35) -> list[Recall]:
        items = sorted(items, key=lambda r: r.score, reverse=True)
        chosen: list[Recall] = []
        vecs: list[np.ndarray] = []
        while items and len(chosen) < k:
            best_i, best_val = 0, -1e9
            for i, item in enumerate(items):
                vec = self.store.vector(item.record.id)
                redundancy = max((float(vec @ v) for v in vecs), default=0.0) if vec is not None else 0.0
                val = item.score - lam * redundancy
                if val > best_val:
                    best_i, best_val = i, val
            pick = items.pop(best_i)
            chosen.append(pick)
            vec = self.store.vector(pick.record.id)
            if vec is not None:
                vecs.append(vec)
        return chosen

    def format_recalls(self, recalls: list[Recall], now: Optional[float] = None) -> list[str]:
        now = now or time.time()
        return [f"({humanize_age(now - r.record.created)}) {r.record.text}" for r in recalls]

    # ------------------------------------------------------------------ viewers
    def seen_user(self, name: str, platform: str = "", support: float = 0.0) -> tuple[UserProfile, bool]:
        return self.store.seen_user(name, platform, self.session_id, support=support)

    def viewer_note(self, name: str, now: Optional[float] = None) -> str:
        profile = self.store.user(name)
        if profile is None:
            return f"{name}: first message ever."
        now = now or time.time()
        if profile.messages <= 1 and profile.sessions <= 1:
            status = "first time chatting"
        elif profile.sessions <= 1:
            status = f"new today ({profile.messages} messages)"
        else:
            status = (f"regular since {time.strftime('%b %d', time.localtime(profile.first_seen))}, "
                      f"{profile.messages} messages over {profile.sessions} streams")
        if profile.support:
            status += ", has supported the stream"
        facts = [r.text for r in self.store.by_subject(name, limit=4)]
        note = f"{name}: {status}"
        if profile.notes:
            note += f"; notes: {profile.notes}"
        if facts:
            note += "; known: " + "; ".join(facts)
        return note

    async def extract_facts(self, user: str, text: str) -> list[str]:
        if not _DISCLOSURE.search(text):
            return []
        facts: list[str] = []
        if self.llm is not None and self.cfg.extract_facts:
            try:
                result = await self.llm.complete_json(
                    FACTS_SYSTEM, [{"role": "user", "content": f"{user} said: {text}"}], FACTS_SCHEMA, name="facts"
                )
                for item in (result or {}).get("facts", [])[:3]:
                    fact = str(item.get("fact", "")).strip()
                    if fact:
                        facts.append(fact if fact.lower().startswith(user.lower()) else f"{user}: {fact}")
            except Exception as exc:  # noqa: BLE001
                log.debug("fact extraction failed: %s", exc)
        if not facts and (self.llm is None or not self.cfg.extract_facts):
            facts = [f"{user} said: {text.strip()[:200]}"]
        for fact in facts:
            await self.remember(fact, kind="fact", importance=6, speaker=user, subject=user)
        return facts

    # ------------------------------------------------------------------ exchange hook
    def observe(self, *, speaker: str, said: str, reply: str, character: str, kind: str = "chat",
                support: float = 0.0, creator: bool = False, first_time: bool = False) -> None:
        """Record an exchange in long-term memory (non-blocking)."""
        if not reply.strip() and not said.strip():
            return
        text = f'{speaker} said "{said.strip()[:240]}" and {character} replied "{reply.strip()[:240]}"'
        importance = estimate_importance(said, support=support, creator=creator, first_time=first_time)
        self._spawn(self.remember(text, kind="episode", importance=importance, speaker=speaker,
                                  subject=speaker if kind in ("chat", "voice", "event") else ""))
        if kind in ("chat", "voice"):
            self._spawn(self.extract_facts(speaker, said))

    async def reflect(self) -> list[str]:
        recent = self.store.recent("episode", limit=40)
        if len(recent) < 5:
            return []
        listing = "\n".join(f"- {r.text}" for r in reversed(recent))
        insights: list[tuple[str, int]] = []
        if self.llm is not None:
            try:
                result = await self.llm.complete_json(
                    REFLECT_SYSTEM.format(name=self.character),
                    [{"role": "user", "content": f"Recent memories:\n{listing}"}],
                    REFLECT_SCHEMA,
                    name="reflection",
                )
                for item in (result or {}).get("insights", [])[:5]:
                    text = str(item.get("text", "")).strip()
                    if text:
                        insights.append((text, int(item.get("importance", 6) or 6)))
            except Exception as exc:  # noqa: BLE001
                log.debug("reflection failed: %s", exc)
        for text, imp in insights:
            await self.remember(f"Insight: {text}", kind="reflection", importance=max(1, min(10, imp)))
        return [t for t, _ in insights]

    async def end_session(self) -> dict:
        """Summarise the stream and write a diary entry. Call once at shutdown."""
        await self.drain(timeout=15)
        episodes = self.store.recent("episode", limit=60, session=self.session_id)
        summary, diary = self.summary, ""
        if episodes and self.llm is not None:
            listing = "\n".join(f"- {r.text}" for r in reversed(episodes))
            try:
                summary = (await self.llm.complete(
                    SUMMARIZE_SYSTEM.format(name=self.character),
                    [{"role": "user", "content": f"Summarise today's stream from these memories:\n{listing}"}],
                    max_tokens=400,
                )).strip()
                diary = (await self.llm.complete(
                    DIARY_SYSTEM.format(name=self.character),
                    [{"role": "user", "content": f"Diary time. Memories:\n{listing}"}],
                    max_tokens=300,
                )).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("end-of-stream summary failed: %s", exc)
        if summary:
            await self.remember(f"Stream on {time.strftime('%b %d')}: {summary}", kind="summary", importance=7)
        if diary:
            await self.remember(f"{self.character}'s diary: {diary}", kind="diary", importance=6)
        self.store.end_session(self.session_id, summary, diary)
        return {"summary": summary, "diary": diary, "episodes": len(episodes)}

    def last_stream_recap(self) -> str:
        sessions = [s for s in self.store.last_sessions(2) if s["id"] != self.session_id]
        if not sessions or not sessions[0].get("summary"):
            return ""
        return sessions[0]["summary"][:400]

    async def aclose(self) -> None:
        await self.drain(timeout=5)
        for task in list(self._bg):
            task.cancel()
        await self.embedder.aclose()
        self.store.close()
