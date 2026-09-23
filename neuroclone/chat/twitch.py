"""Twitch chat over IRC-on-WebSocket. Reading works anonymously; sending needs an OAuth token."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, Union

import aiohttp

from ..config import TwitchConfig
from ..events import ChatMessage, StreamEvent

log = logging.getLogger(__name__)

URL = "wss://irc-ws.chat.twitch.tv:443"
_UNESCAPE = {"\\s": " ", "\\:": ";", "\\\\": "\\", "\\r": "\r", "\\n": "\n"}

Sink = Callable[[Union[ChatMessage, StreamEvent]], Union[None, Awaitable[None]]]


def _unescape(value: str) -> str:
    out, i = [], 0
    while i < len(value):
        pair = value[i : i + 2]
        if pair in _UNESCAPE:
            out.append(_UNESCAPE[pair])
            i += 2
        elif value[i] == "\\":
            i += 1
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def parse_line(line: str) -> dict:
    """Parse one IRCv3 line into {tags, prefix, nick, command, params, trailing}."""
    tags: dict[str, str] = {}
    rest = line.rstrip("\r\n")
    if rest.startswith("@"):
        raw, _, rest = rest[1:].partition(" ")
        for item in raw.split(";"):
            key, _, value = item.partition("=")
            tags[key] = _unescape(value)
    prefix = ""
    if rest.startswith(":"):
        prefix, _, rest = rest[1:].partition(" ")
    trailing = ""
    if " :" in rest:
        rest, _, trailing = rest.partition(" :")
    elif rest.startswith(":"):
        trailing, rest = rest[1:], ""
    parts = rest.split()
    return {
        "tags": tags,
        "prefix": prefix,
        "nick": prefix.split("!", 1)[0] if "!" in prefix else "",
        "command": parts[0] if parts else "",
        "params": parts[1:],
        "trailing": trailing,
    }


def _badges(tags: dict) -> set[str]:
    return {b.split("/", 1)[0] for b in tags.get("badges", "").split(",") if b}


def to_events(msg: dict) -> list[Union[ChatMessage, StreamEvent]]:
    tags, cmd = msg["tags"], msg["command"]
    user = tags.get("display-name") or msg["nick"] or tags.get("login", "")
    if cmd == "PRIVMSG":
        text = msg["trailing"]
        if text.startswith("\x01ACTION ") and text.endswith("\x01"):
            text = text[8:-1]
        bits = int(tags.get("bits", "0") or 0)
        chat = ChatMessage(user=user, text=text, platform="twitch", user_id=tags.get("user-id", ""),
                           badges=_badges(tags), bits=bits, first_time=tags.get("first-msg") == "1")
        out: list = [chat]
        if bits:
            out.append(StreamEvent("bits", user, amount=bits, message=text))
        return out
    if cmd == "USERNOTICE":
        kind = tags.get("msg-id", "")
        text = msg["trailing"]
        if kind in ("sub", "resub"):
            months = float(tags.get("msg-param-cumulative-months", "1") or 1)
            return [StreamEvent(kind, user, amount=months, message=text)]
        if kind == "subgift":
            return [StreamEvent("gift", user, amount=1, message=tags.get("msg-param-recipient-display-name", ""))]
        if kind == "submysterygift":
            return [StreamEvent("gift", user, amount=float(tags.get("msg-param-mass-gift-count", "1") or 1))]
        if kind == "raid":
            raider = tags.get("msg-param-displayName") or user
            return [StreamEvent("raid", raider, amount=float(tags.get("msg-param-viewerCount", "0") or 0))]
    return []


class TwitchChat:
    def __init__(self, cfg: TwitchConfig, sink: Sink, ignore_users: Optional[list] = None, url: str = URL) -> None:
        self.cfg = cfg
        self.url = url
        self.sink = sink
        self.ignore = {u.lower() for u in (ignore_users or [])}
        self.channel = cfg.channel.lstrip("#").lower()
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._closing = False
        self.connected = False

    @property
    def can_send(self) -> bool:
        return bool(self.cfg.oauth_token and self.cfg.nick)

    async def _emit(self, event) -> None:
        user = getattr(event, "user", "").lower()
        if user in self.ignore:
            return
        result = self.sink(event)
        if asyncio.iscoroutine(result):
            await result

    async def send(self, text: str) -> None:
        if self._ws is None or self._ws.closed or not self.can_send:
            return
        await self._ws.send_str(f"PRIVMSG #{self.channel} :{text[:450]}")

    async def run(self) -> None:
        if not self.channel:
            return
        delay = 1.0
        async with aiohttp.ClientSession(trust_env=True) as session:
            while not self._closing:
                try:
                    async with session.ws_connect(self.url, heartbeat=60) as ws:
                        self._ws = ws
                        token = self.cfg.oauth_token
                        if token and self.cfg.nick:
                            await ws.send_str(f"PASS {token if token.startswith('oauth:') else 'oauth:' + token}")
                            await ws.send_str(f"NICK {self.cfg.nick.lower()}")
                        else:
                            await ws.send_str(f"NICK justinfan{random.randint(10000, 99999)}")
                        await ws.send_str("CAP REQ :twitch.tv/tags twitch.tv/commands")
                        await ws.send_str(f"JOIN #{self.channel}")
                        self.connected = True
                        log.info("Twitch chat connected to #%s (%s)", self.channel,
                                 "read/write" if self.can_send else "read-only")
                        delay = 1.0
                        async for frame in ws:
                            if frame.type != aiohttp.WSMsgType.TEXT:
                                break
                            reconnect = False
                            for line in frame.data.split("\r\n"):
                                if not line:
                                    continue
                                if line.startswith("PING"):
                                    await ws.send_str("PONG" + line[4:])
                                    continue
                                parsed = parse_line(line)
                                if parsed["command"] == "RECONNECT":
                                    reconnect = True
                                    break
                                if parsed["command"] == "NOTICE" and "authentication failed" in parsed["trailing"].lower():
                                    log.error("Twitch rejected the OAuth token (check chat.twitch.oauth_token and nick)")
                                for event in to_events(parsed):
                                    await self._emit(event)
                            if reconnect:
                                break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("Twitch chat error: %s (reconnecting in %.0fs)", exc, delay)
                finally:
                    self.connected = False
                    self._ws = None
                if not self._closing:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

    async def aclose(self) -> None:
        self._closing = True
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
