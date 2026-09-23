"""Event types flowing into the conductor, and a small pub/sub bus for UI and logging."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

log = logging.getLogger(__name__)
_ids = itertools.count(1)


def _next_id() -> int:
    return next(_ids)


@dataclass
class ChatMessage:
    user: str
    text: str
    platform: str = "console"  # twitch | youtube | console | dashboard
    user_id: str = ""
    badges: set = field(default_factory=set)  # broadcaster, moderator, vip, subscriber, member...
    bits: int = 0
    first_time: bool = False
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=_next_id)

    @property
    def is_mod(self) -> bool:
        return bool(self.badges & {"moderator", "broadcaster"})

    @property
    def is_sub(self) -> bool:
        return bool(self.badges & {"subscriber", "member", "founder"})


@dataclass
class StreamEvent:
    kind: str  # sub | resub | gift | raid | bits | superchat | member | follow | donation
    user: str
    amount: float = 0.0  # months, gift count, viewers, bits, currency amount
    message: str = ""
    platform: str = "twitch"
    display_amount: str = ""
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=_next_id)


@dataclass
class VoiceTranscript:
    speaker: str
    text: str
    source: str = "mic"  # mic | game_vc | dashboard
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=_next_id)


@dataclass
class SpeechStarted:
    """A human started talking (used for barge-in)."""

    speaker: str
    ts: float = field(default_factory=time.time)


@dataclass
class GameContext:
    game: str
    message: str
    silent: bool = True
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=_next_id)


@dataclass
class ActionForce:
    game: str
    query: str
    action_names: list
    state: str = ""
    ephemeral_context: bool = False
    priority: str = "low"  # low | medium | high | critical
    ts: float = field(default_factory=time.time)
    id: int = field(default_factory=_next_id)


@dataclass
class VisionObservation:
    description: str
    source: str = "screen"
    ts: float = field(default_factory=time.time)


@dataclass
class ModeratorCommand:
    command: str  # pause | resume | skip | say | topic | creator | chat | block | unblock | mute | unmute | twin | look | reset_force
    args: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Event = Union[
    ChatMessage,
    StreamEvent,
    VoiceTranscript,
    SpeechStarted,
    GameContext,
    ActionForce,
    VisionObservation,
    ModeratorCommand,
]

Listener = Callable[[str, dict], Union[None, Awaitable[None]]]


class EventBus:
    """Fan-out of (topic, payload) notifications to the dashboard, loggers and tests.

    Publishing never blocks the caller and never raises: listener errors are logged.
    """

    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self._tasks: set[asyncio.Task] = set()
        self.history: list[tuple[str, dict]] = []
        self.history_limit = 500

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def publish(self, topic: str, payload: Optional[dict] = None) -> None:
        payload = dict(payload or {})
        payload.setdefault("ts", time.time())
        self.history.append((topic, payload))
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        for listener in list(self._listeners):
            try:
                result = listener(topic, payload)
                if inspect.isawaitable(result):
                    task = asyncio.ensure_future(result)
                    self._tasks.add(task)
                    task.add_done_callback(self._done)
            except Exception:  # noqa: BLE001 - a broken listener must not break the stream
                log.exception("event listener failed for topic %s", topic)

    def _done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("async event listener failed: %r", task.exception())

    def recent(self, topic: Optional[str] = None, limit: int = 50) -> list[tuple[str, dict]]:
        items = [h for h in self.history if topic is None or h[0] == topic]
        return items[-limit:]


def event_to_dict(event: Any) -> dict:
    data = asdict(event)
    for key, value in list(data.items()):
        if isinstance(value, set):
            data[key] = sorted(value)
    data["type"] = type(event).__name__
    return data
