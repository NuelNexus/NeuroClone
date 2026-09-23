"""A Neuro SDK-compatible game server (github.com/VedalAI/neuro-sdk, API/SPECIFICATION.md).

Games connect exactly as they would to Neuro-sama (default ``ws://localhost:8000``, set via
``NEURO_SDK_WS_URL`` by the official integrations). Implemented commands:

  game -> us : startup, context, actions/register, actions/unregister, actions/force,
               action/result, shutdown/ready
  us -> game : startup (ack with session/character metadata), action, speech_finished

Unknown commands and malformed messages are silently ignored, as the spec requires.
Game context survives disconnects; ``startup`` clears only the game's registered actions.
The optional voice side-channel lives at ``/game/<name>/voice`` (see voice_chat.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import unquote

from aiohttp import WSMsgType, web

from ..config import GamesConfig
from ..events import ActionForce, EventBus, GameContext

log = logging.getLogger(__name__)

PRIORITIES = ("low", "medium", "high", "critical")


@dataclass
class ActionDef:
    name: str
    description: str = ""
    schema: Optional[dict] = None


@dataclass
class ActionResult:
    success: bool
    message: str = ""
    timed_out: bool = False


@dataclass
class GameSession:
    game: str
    session_id: str
    ws: Optional[web.WebSocketResponse] = None
    actions: dict = field(default_factory=dict)  # name -> ActionDef
    context: deque = field(default_factory=lambda: deque(maxlen=40))  # (ts, text)
    pending_force: Optional[ActionForce] = None
    connected_at: float = field(default_factory=time.time)
    last_context_at: float = 0.0

    @property
    def connected(self) -> bool:
        return self.ws is not None and not self.ws.closed


class NeuroApiServer:
    def __init__(
        self,
        cfg: GamesConfig,
        character_id: str,
        display_name: str,
        submit: Callable[[object], None],
        bus: Optional[EventBus] = None,
    ) -> None:
        self.cfg = cfg
        self.character_id = character_id
        self.display_name = display_name
        self.submit = submit
        self.bus = bus or EventBus()
        self.sessions: dict[str, GameSession] = {}
        self._results: dict[str, asyncio.Future] = {}
        self._runner: Optional[web.AppRunner] = None
        self.voice = None  # optional VoiceChatHub
        self.port = cfg.port

    # ------------------------------------------------------------------ server lifecycle
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/{tail:.*}", self._route)
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.cfg.host, self.cfg.port)
        await site.start()
        addresses = self._runner.addresses  # resolves port 0 (used by tests)
        if addresses:
            self.port = addresses[0][1]
        log.info("Neuro API server listening on ws://%s:%d (character %s)", self.cfg.host, self.port, self.display_name)

    async def stop(self) -> None:
        for fut in self._results.values():
            if not fut.done():
                fut.set_result(ActionResult(False, "server shutting down"))
        for session in self.sessions.values():
            if session.connected:
                await session.ws.close()
        if self._runner is not None:
            await self._runner.cleanup()

    async def _route(self, request: web.Request) -> web.StreamResponse:
        path = request.path.rstrip("/")
        if path.startswith("/game/") and path.endswith("/voice"):
            game = unquote(path[len("/game/") : -len("/voice")])
            if self.voice is None or not self.cfg.voice_chat:
                raise web.HTTPNotFound()
            return await self.voice.handle(request, game)
        if request.headers.get("Upgrade", "").lower() != "websocket":
            return web.json_response({"service": "neuroclone-neuro-api", "character": self.display_name,
                                      "games": self.status()})
        return await self._game_ws(request)

    # ------------------------------------------------------------------ game connection
    async def _game_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        bound: set[str] = set()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    raw = msg.data
                elif msg.type == WSMsgType.BINARY:
                    try:
                        raw = msg.data.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                else:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                command, game = payload.get("command"), payload.get("game")
                if not isinstance(command, str) or not isinstance(game, str) or not game:
                    continue
                data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                session = self._bind(game, ws)
                bound.add(game)
                try:
                    await self._dispatch(session, command, data)
                except Exception:  # noqa: BLE001 - a bad message must not drop the connection
                    log.exception("error handling %s from %s", command, game)
        finally:
            for game in bound:
                session = self.sessions.get(game)
                if session is not None and session.ws is ws:
                    session.ws = None
                    session.pending_force = None
                    self.bus.publish("game", {"game": game, "event": "disconnected"})
        return ws

    def _bind(self, game: str, ws: web.WebSocketResponse) -> GameSession:
        session = self.sessions.get(game)
        if session is None:
            session = GameSession(game=game, session_id=uuid.uuid4().hex,
                                  context=deque(maxlen=self.cfg.context_log_size))
            self.sessions[game] = session
        if session.ws is not ws:
            session.ws = ws
            self.bus.publish("game", {"game": game, "event": "connected"})
        return session

    async def _send(self, session: GameSession, command: str, data: Optional[dict] = None) -> bool:
        if not session.connected:
            return False
        message = {"command": command}
        if data is not None:
            message["data"] = data
        try:
            await session.ws.send_str(json.dumps(message))
            return True
        except (ConnectionError, RuntimeError) as exc:
            log.debug("send to %s failed: %s", session.game, exc)
            return False

    async def _dispatch(self, session: GameSession, command: str, data: dict) -> None:
        if command == "startup":
            session.actions.clear()
            session.pending_force = None
            await self._send(session, "startup", {"session": {
                "sessionId": session.session_id, "characterId": self.character_id, "displayName": self.display_name,
            }})
            self.bus.publish("game", {"game": session.game, "event": "startup"})
        elif command == "context":
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                silent = bool(data.get("silent", True))
                self.add_context(session.game, message)
                self.submit(GameContext(game=session.game, message=message, silent=silent))
        elif command == "actions/register":
            added = []
            for item in data.get("actions") or []:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
                    continue
                schema = item.get("schema")
                schema = schema if isinstance(schema, dict) and schema else None
                session.actions[item["name"]] = ActionDef(item["name"], str(item.get("description", "")), schema)
                added.append(item["name"])
            self.bus.publish("game", {"game": session.game, "event": "register", "actions": added})
        elif command == "actions/unregister":
            for name in data.get("action_names") or []:
                session.actions.pop(name, None)
            force = session.pending_force
            if force is not None and not any(n in session.actions for n in force.action_names):
                session.pending_force = None  # the force refers only to dead actions: ignore it
            self.bus.publish("game", {"game": session.game, "event": "unregister", "actions": data.get("action_names")})
        elif command == "actions/force":
            names = [n for n in data.get("action_names") or [] if isinstance(n, str) and n in session.actions]
            query = data.get("query")
            if not names or not isinstance(query, str):
                return
            priority = data.get("priority", "low")
            force = ActionForce(
                game=session.game,
                query=query,
                action_names=names,
                state=str(data.get("state") or ""),
                ephemeral_context=bool(data.get("ephemeral_context", False)),
                priority=priority if priority in PRIORITIES else "low",
            )
            session.pending_force = force
            self.bus.publish("game", {"game": session.game, "event": "force", "actions": names, "priority": force.priority})
            self.submit(force)
        elif command == "action/result":
            fut = self._results.get(str(data.get("id", "")))
            if fut is not None and not fut.done():
                fut.set_result(ActionResult(bool(data.get("success", False)), str(data.get("message") or "")))
        elif command == "shutdown/ready":
            self.bus.publish("game", {"game": session.game, "event": "shutdown_ready"})
        # anything else: ignored, per spec

    # ------------------------------------------------------------------ API for the brain
    async def execute(self, game: str, name: str, data: Optional[str] = None,
                      timeout: Optional[float] = None) -> ActionResult:
        session = self.sessions.get(game)
        if session is None or not session.connected:
            return ActionResult(False, f"{game} is not connected")
        action_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._results[action_id] = fut
        payload = {"id": action_id, "name": name}
        if data is not None:
            payload["data"] = data
        self.bus.publish("game", {"game": game, "event": "action", "name": name, "data": data})
        try:
            if not await self._send(session, "action", payload):
                return ActionResult(False, "could not reach the game")
            result = await asyncio.wait_for(fut, timeout or self.cfg.action_timeout_s)
        except asyncio.TimeoutError:
            result = ActionResult(False, "The game did not answer in time.", timed_out=True)
        finally:
            self._results.pop(action_id, None)
        self.bus.publish("game", {"game": game, "event": "result", "name": name, "success": result.success,
                                  "message": result.message})
        return result

    async def speech_finished(self, is_final: bool, cancelled: bool = False, reason: Optional[str] = None) -> None:
        data: dict = {"isFinal": is_final}
        if cancelled:
            data["cancelled"] = True
            if reason:
                data["reason"] = reason
        for session in list(self.sessions.values()):
            await self._send(session, "speech_finished", data)

    def add_context(self, game: str, text: str) -> None:
        session = self.sessions.get(game)
        if session is None:
            session = GameSession(game=game, session_id=uuid.uuid4().hex, context=deque(maxlen=self.cfg.context_log_size))
            self.sessions[game] = session
        session.context.append((time.time(), text))
        session.last_context_at = time.time()

    def context_log(self, game: str, limit: int = 15) -> list[str]:
        session = self.sessions.get(game)
        return [text for _, text in list(session.context)[-limit:]] if session else []

    def actions(self, game: str) -> list[ActionDef]:
        session = self.sessions.get(game)
        return list(session.actions.values()) if session else []

    def action(self, game: str, name: str) -> Optional[ActionDef]:
        session = self.sessions.get(game)
        return session.actions.get(name) if session else None

    def pending_force(self, game: str) -> Optional[ActionForce]:
        session = self.sessions.get(game)
        return session.pending_force if session else None

    def clear_force(self, game: str, force: ActionForce) -> None:
        session = self.sessions.get(game)
        if session is not None and session.pending_force is force:
            session.pending_force = None

    def active_games(self) -> list[str]:
        return [g for g, s in self.sessions.items() if s.connected]

    def status(self) -> list[dict]:
        return [{"game": s.game, "connected": s.connected, "actions": sorted(s.actions),
                 "pending_force": bool(s.pending_force)} for s in self.sessions.values()]
