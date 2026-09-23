"""Terminal input for local testing: type as viewers, as the creator, or as a moderator."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Callable, Optional

from ..events import ChatMessage, GameContext, ModeratorCommand, StreamEvent, VoiceTranscript

log = logging.getLogger(__name__)

HELP = """\
Type a line and press Enter:
  hello there              chat as "viewer"
  alice: hello there       chat as alice
  > hey, how's it going?   talk by voice as the creator
  /sub alice 3             alice subscribes (3 months)
  /bits bob 500 nice       bob cheers 500 bits
  /raid carol 42           carol raids with 42 viewers
  /gift dave 5             dave gifts 5 subs
  /game The boss appeared  silent=false game context
  /say <text>              make the character say exactly this (filtered)
  /topic <text>            moderator note: steer the conversation
  /pause  /resume  /skip   moderation controls
  /quit                    stop"""


def parse_console_line(line: str, creator: str = "Creator") -> Optional[object]:
    line = line.strip()
    if not line:
        return None
    if line.startswith(">"):
        return VoiceTranscript(speaker=creator, text=line[1:].strip(), source="console")
    if line.startswith("/"):
        cmd, _, rest = line[1:].partition(" ")
        cmd = cmd.lower()
        args = rest.split()
        try:
            if cmd == "sub":
                return StreamEvent("sub", args[0], amount=float(args[1]) if len(args) > 1 else 1, platform="console")
            if cmd == "bits":
                return StreamEvent("bits", args[0], amount=float(args[1]), message=" ".join(args[2:]), platform="console")
            if cmd == "raid":
                return StreamEvent("raid", args[0], amount=float(args[1]) if len(args) > 1 else 1, platform="console")
            if cmd == "gift":
                return StreamEvent("gift", args[0], amount=float(args[1]) if len(args) > 1 else 1, platform="console")
        except (IndexError, ValueError):
            return ModeratorCommand("help")
        if cmd == "game":
            return GameContext(game="console", message=rest, silent=False)
        if cmd in ("say", "topic"):
            return ModeratorCommand(cmd, {"text": rest})
        if cmd in ("pause", "resume", "skip", "quit", "help", "look", "twin"):
            return ModeratorCommand(cmd, {"text": rest})
        return ModeratorCommand("help")
    user, sep, text = line.partition(":")
    if sep and user and " " not in user.strip() and text.strip():
        return ChatMessage(user=user.strip(), text=text.strip(), platform="console")
    return ChatMessage(user="viewer", text=line, platform="console")


class ConsoleInput:
    def __init__(self, submit: Callable[[object], None], creator: str = "Creator") -> None:
        self.submit = submit
        self.creator = creator
        self._closing = False

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        print(HELP, flush=True)
        while not self._closing:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":  # EOF: Ctrl-D quits now; piped input gets its replies first
                self.submit(ModeratorCommand("quit", {"drain": not sys.stdin.isatty()}))
                return
            event = parse_console_line(line, self.creator)
            if isinstance(event, ModeratorCommand) and event.command == "help":
                print(HELP, flush=True)
                continue
            if event is not None:
                self.submit(event)

    async def aclose(self) -> None:
        self._closing = True
