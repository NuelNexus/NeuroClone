"""The Conductor: decides what the character(s) respond to, when, and how.

An intake task absorbs events instantly (chat, voice, game, moderator); a decision loop picks
the next stimulus by priority and runs one response at a time:

  action force > moderator line/directive > voice (creator, collab, game VC) > stream events
  (batched thanks) > twin banter > chat (or chat vibe) > game context > voluntary game
  action > idle monologue

Game action forces map onto speech interrupts exactly like the Neuro SDK priorities:
critical = stop now, high = after this sentence, medium = soon, low = wait.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .chat.selector import ChatSelector
from .config import Config
from .emotion import EmotionEngine
from .events import (
    ActionForce,
    ChatMessage,
    EventBus,
    GameContext,
    ModeratorCommand,
    SpeechStarted,
    StreamEvent,
    VisionObservation,
    VoiceTranscript,
)
from .games.agent import GameAgent
from .games.neuro_api import NeuroApiServer
from .llm.base import LLM
from .memory.manager import MemoryManager, Turn
from .persona import BitTracker, Persona
from .prompt_builder import PromptBuilder, Stimulus, StreamContext
from .repetition import RepetitionGuard
from .safety.filter import InputFilter
from .speaker import Speaker, SpokenResult
from .transcript import TranscriptLogger

log = logging.getLogger(__name__)

FORCE_INTERRUPTS = {"critical": "now", "high": "after_sentence", "medium": "soon"}


@dataclass
class Character:
    persona: Persona
    speaker: Speaker
    builder: PromptBuilder
    emotion: EmotionEngine
    repetition: RepetitionGuard
    bits: BitTracker
    avatar: object = None  # VTubeStudio or None
    lines: int = 0

    @property
    def name(self) -> str:
        return self.persona.name


@dataclass
class Stats:
    responses: int = 0
    filtered: int = 0
    interrupted: int = 0
    chat_seen: int = 0
    chat_blocked: int = 0
    actions: int = 0
    first_audio: list = field(default_factory=lambda: deque(maxlen=50))

    def as_dict(self) -> dict:
        lat = [x for x in self.first_audio if x is not None]
        return {
            "responses": self.responses, "filtered": self.filtered, "interrupted": self.interrupted,
            "chat_seen": self.chat_seen, "chat_blocked": self.chat_blocked, "actions": self.actions,
            "avg_first_audio_s": round(sum(lat) / len(lat), 3) if lat else None,
        }


class Conductor:
    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLM,
        characters: list[Character],
        selector: ChatSelector,
        input_filter: InputFilter,
        bus: EventBus,
        memory: Optional[MemoryManager] = None,
        games: Optional[NeuroApiServer] = None,
        game_agent: Optional[GameAgent] = None,
        transcripts: Optional[TranscriptLogger] = None,
        vision=None,
        creator: str = "",
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.time,
        on_quit: Optional[Callable[[], None]] = None,
    ) -> None:
        if not characters:
            raise ValueError("at least one character is required")
        self.cfg = cfg
        self.llm = llm
        self.characters = characters
        self.main = characters[0]
        self.selector = selector
        self.input_filter = input_filter
        self.bus = bus
        self.memory = memory
        self.games = games
        self.game_agent = game_agent
        self.transcripts = transcripts
        self.vision = vision
        self.creator = creator or self.main.persona.creator
        self.rng = rng or random.Random()
        self.clock = clock
        self.on_quit = on_quit

        self.inbox: asyncio.Queue = asyncio.Queue()
        self._wake = asyncio.Event()
        self._running = False
        self.paused = False
        self.twin_enabled = len(characters) > 1
        self.pending_forces: dict[str, ActionForce] = {}
        self.pending_says: deque = deque()
        self.pending_directives: deque = deque()
        self.pending_voice: deque = deque(maxlen=20)
        self.pending_events: list[StreamEvent] = []
        self._first_event_ts = 0.0
        self.pending_game: deque = deque(maxlen=10)
        self.pending_banter: Optional[tuple[Character, Character, str]] = None
        self.banter_depth = 0
        self.last_vision: Optional[VisionObservation] = None
        self._last_voluntary: dict[str, float] = {}
        self.started = clock()
        self.last_speech_end = clock()
        self._idle_target = self._next_idle_delay()
        self.stats = Stats()
        self.current: Optional[dict] = None
        self.quit_when_idle = False  # set by a draining quit (e.g. end of piped console input)

    # ------------------------------------------------------------------ plumbing
    def submit(self, event: object) -> None:
        self.inbox.put_nowait(event)

    def _next_idle_delay(self) -> float:
        c = self.cfg.conductor
        return c.idle_after_s + self.rng.uniform(0, max(0.0, c.idle_jitter_s))

    @property
    def speaking(self) -> Optional[Character]:
        return next((c for c in self.characters if c.speaker.speaking), None)

    def interrupt_all(self, mode: str, reason: str) -> None:
        for c in self.characters:
            c.speaker.interrupt(mode, reason)

    def character_by_name(self, name: str) -> Optional[Character]:
        return next((c for c in self.characters if c.name.lower() == (name or "").lower()), None)

    # ------------------------------------------------------------------ run loop
    async def run(self) -> None:
        self._running = True
        intake = asyncio.ensure_future(self._intake())
        ticker = asyncio.ensure_future(self._tick())
        self.bus.publish("status", {"event": "started", "characters": [c.name for c in self.characters]})
        try:
            while self._running:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                if not self._running:
                    break
                if self.quit_when_idle and self.paused:
                    self._quit()
                    break
                if self.paused or self.speaking is not None:
                    continue
                if self.clock() - self.last_speech_end < self.cfg.conductor.reply_gap_s and not self.pending_forces:
                    continue
                stim = self.next_stimulus()
                if self.quit_when_idle and (stim is None or stim.kind in ("idle", "game_voluntary")):
                    if self.inbox.empty() and not self.pending_events:
                        self._quit()
                        break
                    continue  # still draining: no filler talk, wait for the queued work
                if stim is None:
                    continue
                try:
                    await self.perform(stim)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - one bad turn must never end the stream
                    log.exception("turn failed for %s", stim.kind)
                    self.last_speech_end = self.clock()
        finally:
            self._running = False
            for task in (intake, ticker):
                task.cancel()
            for task in (intake, ticker):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    def stop(self) -> None:
        self._running = False
        self.interrupt_all("now", "shutdown")
        self._wake.set()

    def _quit(self) -> None:
        self.stop()
        if self.on_quit:
            self.on_quit()

    async def _intake(self) -> None:
        while True:
            event = await self.inbox.get()
            try:
                await self.handle_event(event)
            except Exception:  # noqa: BLE001
                log.exception("failed to handle %s", type(event).__name__)
            self._wake.set()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            for c in self.characters:
                if c.avatar is not None:
                    c.avatar.set_smile(c.emotion.smile())
            self.bus.publish("state", self.snapshot())
            if self.cfg.conductor.idle_after_s > 0 and self.clock() - self.last_speech_end > self._idle_target:
                self._wake.set()

    # ------------------------------------------------------------------ intake
    async def handle_event(self, event: object) -> None:
        if isinstance(event, ChatMessage):
            self.stats.chat_seen += 1
            verdict = self.input_filter.check(event)
            self.bus.publish("chat", {"user": event.user, "text": event.text, "platform": event.platform,
                                      "allowed": verdict.allowed, "reason": verdict.reason})
            if not verdict.allowed:
                self.stats.chat_blocked += 1
                return
            event.text = verdict.text
            self.selector.add(event)
            if self.memory is not None:
                self.memory.seen_user(event.user, event.platform, support=event.bits / 100 if event.bits else 0)
        elif isinstance(event, StreamEvent):
            if event.message:
                verdict = self.input_filter.check_text(event.message)
                event.message = verdict.text if verdict.allowed else ""
            if not self.pending_events:
                self._first_event_ts = self.clock()
            self.pending_events.append(event)
            for c in self.characters:
                c.emotion.apply_event(event.kind, 1.0 + min(1.0, event.amount / 50))
            if self.memory is not None:
                self.memory.seen_user(event.user, event.platform, support=max(1.0, event.amount))
            self.bus.publish("event", {"kind": event.kind, "user": event.user, "amount": event.amount})
        elif isinstance(event, VoiceTranscript):
            trusted = event.speaker.lower() == self.creator.lower() and event.source in ("mic", "console", "dashboard")
            if not trusted:
                verdict = self.input_filter.check_text(event.text)
                if not verdict.allowed:
                    self.bus.publish("voice", {"speaker": event.speaker, "text": event.text, "allowed": False})
                    return
            self.pending_voice.append(event)
            self.bus.publish("voice", {"speaker": event.speaker, "text": event.text, "allowed": True})
        elif isinstance(event, SpeechStarted):
            if self.cfg.stt.barge_in and self.speaking is not None:
                self.interrupt_all("now", f"{event.speaker} started talking")
        elif isinstance(event, GameContext):
            self.selector.set_topic(f"{event.game} {event.message}")
            if not event.silent and self.cfg.conductor.respond_to_game_context:
                self.pending_game.append(event)
        elif isinstance(event, ActionForce):
            self.pending_forces[event.game] = event
            mode = FORCE_INTERRUPTS.get(event.priority)
            if mode and self.speaking is not None:
                self.interrupt_all(mode, f"{event.game} needs a decision")
        elif isinstance(event, VisionObservation):
            self.last_vision = event
            self.bus.publish("vision", {"description": event.description})
        elif isinstance(event, ModeratorCommand):
            await self.moderate(event)

    async def moderate(self, cmd: ModeratorCommand) -> None:
        text = str(cmd.args.get("text", "")).strip()
        name = cmd.command
        if name == "pause":
            self.paused = True
            self.interrupt_all("now", "paused by moderator")
        elif name == "resume":
            self.paused = False
            self.last_speech_end = self.clock()
        elif name == "skip":
            self.interrupt_all("now", "skipped by moderator")
        elif name == "say" and text:
            self.pending_says.append((text, cmd.args.get("character", "")))
        elif name == "topic" and text:
            self.pending_directives.append(text)
        elif name == "creator" and text:
            self.pending_voice.append(VoiceTranscript(self.creator, text, source="dashboard"))
        elif name == "chat" and text:
            self.submit(ChatMessage(user=str(cmd.args.get("user") or "dashboard"), text=text, platform="dashboard"))
        elif name == "block" and text:
            self.input_filter.blocklist.add(text)
        elif name == "unblock" and text:
            self.input_filter.blocklist.remove(text)
        elif name == "mute" and text:
            self.input_filter.muted.add(text.lower())
        elif name == "unmute" and text:
            self.input_filter.muted.discard(text.lower())
        elif name == "twin":
            self.twin_enabled = len(self.characters) > 1 and text.lower() not in ("off", "0", "false")
        elif name == "look" and self.vision is not None:
            asyncio.ensure_future(self.vision.look())
        elif name == "reset_force":
            self.pending_forces.clear()
        elif name == "quit":
            if cmd.args.get("drain"):
                self.quit_when_idle = True  # finish what is queued first
            else:
                self._quit()
        self.bus.publish("moderation", {"command": name, "text": text})

    # ------------------------------------------------------------------ choosing what to do
    def next_stimulus(self) -> Optional[Stimulus]:
        now = self.clock()
        for game, force in list(self.pending_forces.items()):
            del self.pending_forces[game]
            if self.games is None or self.games.pending_force(game) is force:
                return Stimulus("force", speaker=game, text=force.query, meta={"force": force})
        if self.pending_says:
            text, who = self.pending_says.popleft()
            return Stimulus("say", text=text, target=who)
        if self.pending_directives:
            return Stimulus("director", speaker="director", text=self.pending_directives.popleft())
        if self.pending_voice:
            v = self.pending_voice.popleft()
            return Stimulus("voice", speaker=v.speaker, text=v.text, meta={"source": v.source})
        if self.pending_events and now - self._first_event_ts >= self.cfg.conductor.event_batch_s:
            events, self.pending_events = self.pending_events[:6], self.pending_events[6:]
            if self.pending_events:
                self._first_event_ts = now
            return Stimulus("event", speaker=events[0].user, text=events[0].message, events=events)
        if self.pending_banter is not None and self.twin_enabled:
            responder, other, line = self.pending_banter
            self.pending_banter = None
            return Stimulus("twin", speaker=other.name, text=line, target=responder.name)
        vibe = self.selector.vibe(now)
        if vibe:
            return Stimulus("vibe", speaker="chat", text=vibe)
        cand = self.selector.pick(now)
        if cand is not None:
            self.bus.publish("selected", {"user": cand.msg.user, "text": cand.msg.text, "score": round(cand.score, 2),
                                          "reasons": {k: round(v, 2) for k, v in cand.reasons.items()},
                                          "candidates": self.selector.snapshot()})
            return Stimulus("chat", speaker=cand.msg.user, text=cand.msg.text, msg=cand.msg)
        if self.pending_game:
            g = self.pending_game.popleft()
            return Stimulus("game", speaker=g.game, text=g.message)
        voluntary = self._voluntary_candidate(now)
        if voluntary:
            return Stimulus("game_voluntary", speaker=voluntary)
        if self.cfg.conductor.idle_after_s > 0 and now - self.last_speech_end >= self._idle_target:
            return Stimulus("idle", meta={"seconds": int(now - self.last_speech_end), "idea": self._idle_idea()})
        return None

    def _voluntary_candidate(self, now: float) -> Optional[str]:
        if self.games is None or self.game_agent is None or not self.cfg.games.voluntary_actions:
            return None
        for game in self.games.active_games():
            session = self.games.sessions[game]
            if not session.actions or session.pending_force is not None:
                continue
            last = self._last_voluntary.get(game, 0.0)
            if session.last_context_at > last and now - last >= self.cfg.games.voluntary_interval_s:
                return game
        return None

    def _idle_idea(self) -> str:
        ideas = list(self.main.persona.idle_ideas) or ["start a fun topic"]
        if self.memory is not None and self.rng.random() < 0.3:
            facts = self.memory.store.recent("fact", limit=20) + self.memory.store.recent("reflection", limit=10)
            if facts:
                return f"Bring up this memory naturally: {self.rng.choice(facts).text}"
        return self.rng.choice(ideas)

    def choose_character(self, stim: Stimulus) -> Character:
        if stim.target:
            found = self.character_by_name(stim.target)
            if found:
                return found
        if len(self.characters) == 1 or not self.twin_enabled or stim.kind in ("force", "game", "game_voluntary"):
            return self.main
        text = stim.text.lower()
        for c in self.characters:
            if c.name.lower() in text:
                return c
        if stim.kind == "idle":
            return min(self.characters, key=lambda c: c.lines)
        return self.main if self.rng.random() < 0.7 else self.characters[1]

    # ------------------------------------------------------------------ doing it
    async def perform(self, stim: Stimulus) -> None:
        self.current = {"kind": stim.kind, "speaker": stim.speaker, "text": stim.text[:200]}
        try:
            if stim.kind == "force":
                await self.handle_force(stim.meta["force"])
            elif stim.kind == "game_voluntary":
                await self.voluntary_action(stim.speaker)
            elif stim.kind == "say":
                char = self.choose_character(stim)
                result = await char.speaker.speak_text(stim.text)
                self._record(char, stim, result, prompt=None)
            else:
                await self.respond(stim)
        finally:
            self.current = None
            self.last_speech_end = self.clock()
            self._idle_target = self._next_idle_delay()

    async def build_context(self, char: Character, stim: Stimulus) -> StreamContext:
        now = self.clock()
        ctx = StreamContext(now=now, uptime_s=now - self.started, mood=char.emotion.mood(now).describe(),
                            on_stage=[c.name for c in self.characters] if self.twin_enabled else [char.name])
        if self.games is not None:
            active = self.games.active_games()
            if active:
                ctx.game = active[0]
                ctx.game_events = self.games.context_log(active[0], limit=3)
        if self.last_vision is not None and now - self.last_vision.ts < 90:
            ctx.vision = self.last_vision.description
        if self.memory is not None:
            query = f"{stim.speaker} {stim.text}" if stim.kind != "idle" else str(stim.meta.get("idea", ""))
            try:
                recalls = await self.memory.recall(query, now=now)
                ctx.memories = self.memory.format_recalls(recalls, now)
            except Exception as exc:  # noqa: BLE001
                log.warning("memory recall failed: %s", exc)
            if stim.kind in ("chat", "event", "voice") and stim.speaker and stim.speaker.lower() != self.creator.lower():
                ctx.viewer_note = self.memory.viewer_note(stim.speaker)
            if now - self.started < 600:
                ctx.last_stream = self.memory.last_stream_recap()
        ctx.avoid = char.repetition.overused_phrases()
        ctx.exhausted_bits = char.bits.exhausted(now)
        return ctx

    async def respond(self, stim: Stimulus) -> SpokenResult:
        char = self.choose_character(stim)
        ctx = await self.build_context(char, stim)
        turns = self.memory.recent_turns() if self.memory is not None else []
        summary = self.memory.summary if self.memory is not None else ""
        system, messages = char.builder.build(stim, turns, ctx, summary)
        t_request = time.monotonic()
        tokens = self.llm.stream(system, messages, purpose="chat")
        result = await char.speaker.speak_stream(tokens, uuid.uuid4().hex[:8], t_request=t_request)
        self._record(char, stim, result, prompt=messages)
        return result

    def _record(self, char: Character, stim: Stimulus, result: SpokenResult, prompt: Optional[list]) -> None:
        self.stats.responses += 1
        self.stats.filtered += int(result.filtered)
        self.stats.interrupted += int(result.interrupted)
        self.stats.first_audio.append(result.latency()["first_audio_s"])
        char.lines += 1
        char.bits.record(result.text)
        if self.memory is not None:
            turn = char.builder.stimulus_turn(stim)
            if turn is not None:
                self.memory.add_turn(turn)
            if result.text:
                self.memory.add_turn(Turn(char.name, result.text, "character", character=True))
            if stim.kind in ("chat", "voice", "event") and result.text:
                is_creator = stim.speaker.lower() == self.creator.lower()
                support = sum(e.amount for e in stim.events) if stim.events else 0.0
                self.memory.observe(speaker=stim.speaker or "chat", said=stim.text, reply=result.text,
                                    character=char.name, kind=stim.kind, support=support, creator=is_creator,
                                    first_time=bool(stim.msg and stim.msg.first_time))
        # Twin banter: sometimes the other twin reacts to what was just said.
        if self.twin_enabled and len(self.characters) > 1 and result.text and not result.filtered:
            other = next(c for c in self.characters if c is not char)
            depth = self.banter_depth + 1 if stim.kind == "twin" else 0
            chance = 0.8 if other.name.lower() in result.text.lower() else self.cfg.conductor.banter_chance
            if depth < self.cfg.conductor.max_banter and self.rng.random() < chance:
                self.pending_banter = (other, char, result.text)
                self.banter_depth = depth
            else:
                self.banter_depth = 0
        if self.transcripts is not None:
            self.transcripts.log({
                "character": char.name, "stimulus": {"kind": stim.kind, "speaker": stim.speaker, "text": stim.text},
                "system_fingerprint": TranscriptLogger.fingerprint(char.builder.system), "prompt": prompt,
                "generated": result.generated, "spoken": result.text, "filtered": result.filtered,
                "filter_reason": result.filter_reason, "interrupted": result.interrupted,
                "emotions": result.emotions, "latency": result.latency(), "error": result.error,
                "model": getattr(self.llm, "name", ""),
            })
        self.bus.publish("turn", {"character": char.name, "kind": stim.kind, "speaker": stim.speaker,
                                  "stimulus": stim.text[:300], "reply": result.text, "filtered": result.filtered,
                                  "interrupted": result.interrupted, "latency": result.latency(),
                                  "mood": char.emotion.mood().label})

    # ------------------------------------------------------------------ games
    async def handle_force(self, force: ActionForce) -> None:
        if self.games is None or self.game_agent is None:
            return
        game = force.game
        char = self.main
        if not force.ephemeral_context:
            state = f"State: {force.state.strip()[:600]} | " if force.state else ""
            self.games.add_context(game, f"{state}Task: {force.query}")
        feedback: list[str] = []
        for _ in range(1 + self.cfg.games.max_force_retries):
            if self.games.pending_force(game) is not force:
                return  # superseded or cancelled by the game
            decision = await self.game_agent.decide(game, force, feedback=feedback)
            if decision.action is None:
                feedback.append(decision.error or "no action was chosen")
                continue
            say = asyncio.ensure_future(char.speaker.speak_text(decision.say)) if decision.say else None
            result = await self.games.execute(game, decision.action, decision.data)
            self.stats.actions += 1
            outcome = "succeeded" if result.success else "failed"
            detail = f" ({result.message})" if result.message else ""
            self.games.add_context(game, f"You used {decision.action}{' ' + decision.data if decision.data else ''}: "
                                         f"{outcome}{detail}")
            if say is not None:
                spoken = await say
                self._record(char, Stimulus("game", speaker=game, text=f"(decision) {force.query}"), spoken, None)
            if result.success:
                break
            feedback.append(f"The game rejected {decision.action}: {result.message or 'no reason given'}")
        self.games.clear_force(game, force)

    async def voluntary_action(self, game: str) -> None:
        if self.games is None or self.game_agent is None:
            return
        self._last_voluntary[game] = self.clock()
        decision = await self.game_agent.decide(game, None, allow_none=True)
        char = self.main
        say = asyncio.ensure_future(char.speaker.speak_text(decision.say)) if decision.say else None
        if decision.action:
            result = await self.games.execute(game, decision.action, decision.data)
            self.stats.actions += 1
            self.games.add_context(game, f"You chose to use {decision.action}: "
                                         f"{'succeeded' if result.success else 'failed'}"
                                         f"{' (' + result.message + ')' if result.message else ''}")
        if say is not None:
            spoken = await say
            self._record(char, Stimulus("game", speaker=game, text="(your own move)"), spoken, None)

    # ------------------------------------------------------------------ dashboard
    def snapshot(self) -> dict:
        speaking = self.speaking
        return {
            "paused": self.paused,
            "speaking": speaking.name if speaking else None,
            "current": self.current,
            "twin": self.twin_enabled,
            "characters": [{"name": c.name, "mood": c.emotion.mood().label, "emotion": c.emotion.emotion,
                            "valence": round(c.emotion.valence, 2), "arousal": round(c.emotion.arousal, 2),
                            "lines": c.lines,
                            "avatar": c.avatar.status() if c.avatar is not None else None} for c in self.characters],
            "pending": {"chat": self.selector.pending(), "voice": len(self.pending_voice),
                        "events": len(self.pending_events), "forces": len(self.pending_forces)},
            "games": self.games.status() if self.games is not None else [],
            "stats": self.stats.as_dict(),
            "uptime_s": round(self.clock() - self.started),
        }
