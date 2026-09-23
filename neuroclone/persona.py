"""Persona cards: YAML character definitions rendered into a stable system prompt."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Optional

import yaml

from .speech.text import DEFAULT_EMOTIONS


class PersonaError(ValueError):
    pass


@dataclass
class RunningBit:
    name: str
    description: str
    keywords: list = field(default_factory=list)
    max_per_hour: int = 3


@dataclass
class Example:
    user: str
    reply: str
    speaker: str = "viewer"  # viewer | creator | twin


@dataclass
class Persona:
    id: str
    name: str
    creator: str = "the developer"
    audience_name: str = "chat"
    platform: str = "Twitch"
    twin: str = ""
    tagline: str = ""
    identity: str = ""
    personality: list = field(default_factory=list)
    speech_style: list = field(default_factory=list)
    humor: list = field(default_factory=list)
    likes: list = field(default_factory=list)
    dislikes: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)
    running_bits: list = field(default_factory=list)
    lore: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    deflections: list = field(default_factory=list)
    idle_ideas: list = field(default_factory=list)
    voice: dict = field(default_factory=dict)
    emotions: list = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    avatar_hotkeys: dict = field(default_factory=dict)
    source: str = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "Persona":
        if not isinstance(data, dict) or not data.get("name"):
            raise PersonaError(f"{source or 'persona'}: a persona needs at least a name")
        data = dict(data)
        data.setdefault("id", re.sub(r"\W+", "_", data["name"].lower()))
        bits = [b if isinstance(b, RunningBit) else RunningBit(**b) for b in data.pop("running_bits", []) or []]
        examples = [e if isinstance(e, Example) else Example(**e) for e in data.pop("examples", []) or []]
        known = set(cls.__dataclass_fields__) - {"running_bits", "examples", "source"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise PersonaError(f"{source or data['id']}: unknown persona keys {unknown}")
        persona = cls(**data, running_bits=bits, examples=examples, source=source)
        persona.emotions = [e.lower() for e in persona.emotions] or list(DEFAULT_EMOTIONS)
        return persona

    def with_creator(self, creator: str) -> "Persona":
        """Rename the creator everywhere (config ``creator:`` override)."""
        if not creator or creator == self.creator:
            return self
        old = self.creator
        swap = lambda s: s.replace(old, creator) if isinstance(s, str) else s  # noqa: E731
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for key in ("identity", "tagline"):
            data[key] = swap(data[key])
        for key in ("personality", "speech_style", "humor", "likes", "dislikes", "lore", "boundaries",
                    "deflections", "idle_ideas"):
            data[key] = [swap(x) for x in data[key]]
        data["relationships"] = {swap(k): swap(v) for k, v in data["relationships"].items()}
        data["running_bits"] = [RunningBit(b.name, swap(b.description), b.keywords, b.max_per_hour)
                                for b in data["running_bits"]]
        data["examples"] = [Example(swap(e.user), swap(e.reply), e.speaker) for e in data["examples"]]
        data["creator"] = creator
        return Persona(**data)

    # ------------------------------------------------------------------ prompt
    def system_prompt(self, twin: Optional["Persona"] = None) -> str:
        from .prompts import render_system_prompt

        return render_system_prompt(self, twin)

    def bits_used(self, text: str) -> list[str]:
        low = text.lower()
        return [b.name for b in self.running_bits if any(k.lower() in low for k in b.keywords)]

    def voice_for(self, provider: str) -> dict:
        return dict(self.voice.get(provider, {}) or {})


class BitTracker:
    """Keeps running bits fresh: counts uses per rolling hour and reports exhausted ones."""

    def __init__(self, persona: Persona) -> None:
        self.persona = persona
        self.uses: dict[str, list[float]] = {}

    def record(self, text: str, now: Optional[float] = None) -> list[str]:
        now = now or time.time()
        used = self.persona.bits_used(text)
        for name in used:
            self.uses.setdefault(name, []).append(now)
        return used

    def exhausted(self, now: Optional[float] = None) -> list[str]:
        now = now or time.time()
        out = []
        for bit in self.persona.running_bits:
            recent = [t for t in self.uses.get(bit.name, []) if now - t < 3600]
            self.uses[bit.name] = recent
            if len(recent) >= bit.max_per_hour:
                out.append(bit.name.replace("_", " "))
        return out


def _packaged_persona(name: str) -> Optional[str]:
    try:
        path = resources.files("neuroclone.data").joinpath("personas", f"{name}.yaml")
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def load_persona(ref: str, personas_dir: str | Path = "config/personas", creator: str = "") -> Persona:
    """Load by path (``my/char.yaml``) or by id (looked up in personas_dir, then the packaged set)."""
    if not ref:
        raise PersonaError("no persona given")
    candidates = [Path(ref)] if ref.endswith((".yaml", ".yml")) else []
    candidates += [Path(personas_dir) / f"{ref}.yaml", Path(personas_dir) / f"{ref}.yml"]
    for path in candidates:
        if path.is_file():
            persona = Persona.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
            return persona.with_creator(creator)
    text = _packaged_persona(ref)
    if text is None:
        raise PersonaError(f"persona {ref!r} not found (looked in {personas_dir} and the built-in set)")
    return Persona.from_dict(yaml.safe_load(text), f"builtin:{ref}").with_creator(creator)


def list_builtin_personas() -> list[str]:
    try:
        folder = resources.files("neuroclone.data").joinpath("personas")
        return sorted(p.name.rsplit(".", 1)[0] for p in folder.iterdir() if p.name.endswith(".yaml"))
    except (FileNotFoundError, ModuleNotFoundError):
        return []
