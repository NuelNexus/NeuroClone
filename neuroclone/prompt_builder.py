"""Builds (system, messages) for each response.

- The system prompt is the persona's stable prompt (cache-friendly).
- History is rendered from the responding character's perspective: her own lines are
  ``assistant`` turns; everything else (chat, voices, events, her twin) is ``user`` content
  wrapped in tags. Consecutive same-role turns are merged and the list always starts with
  ``user``, which keeps strict chat templates happy.
- Volatile context (time, mood, memories, game state) goes into the final user turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .events import ChatMessage, StreamEvent
from .memory.manager import Turn
from .persona import Persona
from .prompts import IDLE_INSTRUCTION, REPLY_INSTRUCTION


def esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def attr(text: str) -> str:
    return esc(text).replace('"', "&quot;")


@dataclass
class Stimulus:
    kind: str  # chat | vibe | voice | event | twin | idle | game | director | force | say | game_voluntary
    speaker: str = ""
    text: str = ""
    msg: Optional[ChatMessage] = None
    events: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    target: str = ""  # which character should answer (name), if decided
    ts: float = field(default_factory=time.time)


@dataclass
class StreamContext:
    now: float = field(default_factory=time.time)
    uptime_s: float = 0.0
    mood: str = ""
    on_stage: list = field(default_factory=list)
    game: str = ""
    game_events: list = field(default_factory=list)
    memories: list = field(default_factory=list)
    viewer_note: str = ""
    avoid: list = field(default_factory=list)
    exhausted_bits: list = field(default_factory=list)
    vision: str = ""
    last_stream: str = ""
    extra: list = field(default_factory=list)

    def render(self) -> str:
        lt = time.localtime(self.now)
        up_min = int(self.uptime_s // 60)
        uptime = f"{up_min // 60}h {up_min % 60}m" if up_min >= 60 else f"{up_min}m"
        lines = [f"time: {time.strftime('%A %I:%M %p', lt)}, live for {uptime}"]
        if self.mood:
            lines.append(f"your mood: {self.mood}")
        if len(self.on_stage) > 1:
            lines.append(f"on stage: {', '.join(self.on_stage)}")
        if self.game:
            lines.append(f"playing: {self.game}")
            lines += [f"  game: {e}" for e in self.game_events[-3:]]
        if self.vision:
            lines.append(f"on screen: {self.vision}")
        if self.last_stream:
            lines.append(f"last stream recap: {self.last_stream}")
        if self.memories:
            lines.append("memories (may be imperfect):")
            lines += [f"  - {m}" for m in self.memories]
        if self.viewer_note:
            lines.append(f"viewer: {self.viewer_note}")
        if self.avoid:
            lines.append("you've been overusing (say it differently): " + "; ".join(f'"{a}"' for a in self.avoid))
        if self.exhausted_bits:
            lines.append("skip these bits for now: " + ", ".join(self.exhausted_bits))
        lines += self.extra
        return "<stream_context>\n" + "\n".join(esc(l) for l in lines) + "\n</stream_context>"


def render_event(ev: StreamEvent) -> str:
    extra = ""
    if ev.kind in ("sub", "resub", "member") and ev.amount:
        extra = f' months="{int(ev.amount)}"'
    elif ev.kind == "gift":
        extra = f' count="{int(ev.amount)}"'
    elif ev.kind == "raid":
        extra = f' viewers="{int(ev.amount)}"'
    elif ev.kind == "bits":
        extra = f' bits="{int(ev.amount)}"'
    elif ev.kind in ("superchat", "donation"):
        extra = f' amount="{attr(ev.display_amount or str(ev.amount))}"'
    body = esc(ev.message) if ev.message else ""
    return f'<event type="{ev.kind}" user="{attr(ev.user)}"{extra}>{body}</event>'


def render_chat(msg: ChatMessage) -> str:
    attrs = f'user="{attr(msg.user)}"'
    badges = sorted(b for b in msg.badges if b in {"broadcaster", "moderator", "vip", "subscriber", "member", "founder"})
    if badges:
        attrs += f' badges="{",".join(badges)}"'
    if msg.first_time:
        attrs += ' first_time="yes"'
    if msg.bits:
        attrs += f' bits="{msg.bits}"'
    return f"<chat {attrs}>{esc(msg.text)}</chat>"


class PromptBuilder:
    def __init__(self, persona: Persona, twin: Optional[Persona] = None, compact: bool = False) -> None:
        self.persona = persona
        self.twin = twin
        self.system = persona.system_prompt(twin, compact=compact)

    # ------------------------------------------------------------------ history
    def render_turn(self, turn: Turn) -> tuple[str, str]:
        me = self.persona.name
        if turn.character and turn.speaker == me:
            return "assistant", turn.text
        if turn.character:
            return "user", f'<twin name="{attr(turn.speaker)}">{esc(turn.text)}</twin>'
        kind = turn.kind
        if kind == "chat":
            return "user", turn.meta.get("rendered") or f'<chat user="{attr(turn.speaker)}">{esc(turn.text)}</chat>'
        if kind == "voice":
            return "user", f'<voice speaker="{attr(turn.speaker)}">{esc(turn.text)}</voice>'
        if kind == "event":
            return "user", turn.text  # already rendered markup
        if kind == "game":
            return "user", f'<game name="{attr(turn.speaker)}">{esc(turn.text)}</game>'
        if kind == "director":
            return "user", f"<director>{esc(turn.text)}</director>"
        if kind == "vision":
            return "user", f"<vision>{esc(turn.text)}</vision>"
        if kind == "vibe":
            return "user", f"<chat_vibe>{esc(turn.text)}</chat_vibe>"
        return "user", "<idle/>"

    def history_messages(self, turns: list[Turn], summary: str = "") -> list[dict]:
        msgs: list[dict] = []
        if summary:
            msgs.append({"role": "user", "content": f"<earlier_on_stream>{esc(summary)}</earlier_on_stream>"})
        for turn in turns:
            if not turn.text and turn.kind != "idle":
                continue
            role, content = self.render_turn(turn)
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + content
            else:
                msgs.append({"role": role, "content": content})
        if msgs and msgs[0]["role"] == "assistant":
            msgs.insert(0, {"role": "user", "content": "<stream_start/>"})
        return msgs

    # ------------------------------------------------------------------ stimulus
    def render_stimulus(self, stim: Stimulus) -> str:
        kind = stim.kind
        if kind == "chat" and stim.msg is not None:
            return render_chat(stim.msg)
        if kind == "vibe":
            return f"<chat_vibe>{esc(stim.text)}</chat_vibe>\nReact to what chat is collectively doing."
        if kind == "voice":
            return f'<voice speaker="{attr(stim.speaker)}">{esc(stim.text)}</voice>'
        if kind == "event":
            body = "\n".join(render_event(e) for e in stim.events)
            if len(stim.events) > 1:
                body += "\nThank them together in one or two sentences."
            return body
        if kind == "twin":
            return f'<twin name="{attr(stim.speaker)}">{esc(stim.text)}</twin>'
        if kind == "game":
            return f'<game name="{attr(stim.speaker)}">{esc(stim.text)}</game>'
        if kind == "director":
            return f"<director>{esc(stim.text)}</director>"
        if kind == "idle":
            seconds = int(stim.meta.get("seconds", 15))
            idea = stim.meta.get("idea", "anything you like")
            return f'<idle seconds="{seconds}"/>\n' + IDLE_INSTRUCTION.format(seconds=seconds, idea=idea)
        return esc(stim.text)

    def stimulus_turn(self, stim: Stimulus) -> Optional[Turn]:
        """The history entry recorded for a stimulus once it has been answered."""
        if stim.kind == "chat" and stim.msg is not None:
            return Turn(stim.msg.user, stim.msg.text, "chat", meta={"rendered": render_chat(stim.msg)})
        if stim.kind == "event":
            return Turn("stream", "\n".join(render_event(e) for e in stim.events), "event")
        if stim.kind in ("voice", "director", "vibe", "game"):
            return Turn(stim.speaker or stim.kind, stim.text, stim.kind)
        return None  # twin lines are already in history; idle adds nothing

    def build(self, stim: Stimulus, turns: list[Turn], context: StreamContext, summary: str = "") -> tuple[str, list[dict]]:
        if stim.kind == "twin" and turns and turns[-1].character and turns[-1].text == stim.text:
            turns = turns[:-1]  # the line being answered is rendered as the stimulus instead
        messages = self.history_messages(turns, summary)
        final = f"{context.render()}\n{self.render_stimulus(stim)}\n{REPLY_INSTRUCTION.format(name=self.persona.name)}"
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n" + final
        else:
            messages.append({"role": "user", "content": final})
        return self.system, messages
