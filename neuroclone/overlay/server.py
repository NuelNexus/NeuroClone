"""Local web server: moderator dashboard, OBS caption overlay, live event feed, and REST controls.

Binds to 127.0.0.1 by default. Set ``overlay.token`` to require ``?token=...`` (or an
``X-Token`` header) before exposing it on a LAN.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources
from typing import Optional

from aiohttp import WSMsgType, web

from ..config import OverlayConfig
from ..events import EventBus, ModeratorCommand

log = logging.getLogger(__name__)

ALLOWED_COMMANDS = {"pause", "resume", "skip", "say", "topic", "creator", "chat", "block", "unblock", "mute",
                    "unmute", "twin", "look", "reset_force"}


def _static(name: str) -> str:
    return resources.files("neuroclone.overlay").joinpath("static", name).read_text(encoding="utf-8")


class OverlayServer:
    def __init__(self, cfg: OverlayConfig, bus: EventBus, conductor, memory=None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.conductor = conductor
        self.memory = memory
        self.clients: set[web.WebSocketResponse] = set()
        self._runner: Optional[web.AppRunner] = None
        self._unsubscribe = None
        self.port = cfg.port

    def _authorised(self, request: web.Request) -> bool:
        if not self.cfg.token:
            return True
        return request.query.get("token") == self.cfg.token or request.headers.get("X-Token") == self.cfg.token

    @web.middleware
    async def _auth(self, request: web.Request, handler):
        if not self._authorised(request):
            raise web.HTTPUnauthorized(text="missing or wrong token")
        return await handler(request)

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._auth])
        app.router.add_get("/", self._page("dashboard.html"))
        app.router.add_get("/overlay", self._page("overlay.html"))
        app.router.add_get("/ws", self._ws)
        app.router.add_get("/api/state", self._state)
        app.router.add_post("/api/command", self._command)
        app.router.add_get("/api/memory", self._memory)
        return app

    def _page(self, name: str):
        async def handler(_request: web.Request) -> web.Response:
            return web.Response(text=_static(name), content_type="text/html")
        return handler

    async def _state(self, _request: web.Request) -> web.Response:
        return web.json_response(self.conductor.snapshot())

    async def _command(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="expected JSON")
        command = str(body.get("command", ""))
        if command not in ALLOWED_COMMANDS:
            raise web.HTTPBadRequest(text=f"unknown command {command!r}")
        args = body.get("args") if isinstance(body.get("args"), dict) else {}
        self.conductor.submit(ModeratorCommand(command, args))
        return web.json_response({"ok": True})

    async def _memory(self, request: web.Request) -> web.Response:
        if self.memory is None:
            return web.json_response({"enabled": False, "results": []})
        query = request.query.get("q", "").strip()
        results = []
        if query:
            recalls = await self.memory.recall(query, k=int(request.query.get("k", 8)), exclude_recent_s=0)
            results = [{"text": r.record.text, "kind": r.record.kind, "score": round(r.score, 3),
                        "similarity": round(r.similarity, 3), "importance": r.record.importance} for r in recalls]
        return web.json_response({"enabled": True, "stats": self.memory.store.stats(), "results": results})

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            await ws.send_str(json.dumps({"topic": "state", "data": self.conductor.snapshot()}))
            for topic, data in self.bus.recent(limit=80):
                await ws.send_str(json.dumps({"topic": topic, "data": data}, default=str))
            await ws.send_str(json.dumps({"topic": "ready", "data": {}}))  # end of history replay
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        body = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if body.get("command") in ALLOWED_COMMANDS:
                        args = body.get("args") if isinstance(body.get("args"), dict) else {}
                        self.conductor.submit(ModeratorCommand(body["command"], args))
        finally:
            self.clients.discard(ws)
        return ws

    async def _broadcast(self, topic: str, data: dict) -> None:
        if not self.clients:
            return
        text = json.dumps({"topic": topic, "data": data}, default=str)
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_str(text), timeout=1.0)
            except (asyncio.TimeoutError, ConnectionError, RuntimeError):
                self.clients.discard(ws)

    async def start(self) -> None:
        self._unsubscribe = self.bus.subscribe(self._broadcast)
        self._runner = web.AppRunner(self.app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.cfg.host, self.cfg.port)
        await site.start()
        if self._runner.addresses:
            self.port = self._runner.addresses[0][1]
        suffix = f"?token={self.cfg.token}" if self.cfg.token else ""
        log.info("dashboard: http://%s:%d/%s  overlay: http://%s:%d/overlay%s",
                 self.cfg.host, self.port, suffix, self.cfg.host, self.port, suffix)

    async def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
        for ws in list(self.clients):
            await ws.close()
        if self._runner is not None:
            await self._runner.cleanup()
