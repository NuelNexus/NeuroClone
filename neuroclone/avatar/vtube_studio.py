"""VTube Studio public API client.

Lip-sync is injected directly into the ``MouthOpen`` parameter from the TTS audio envelope
(no virtual audio cable), mood drives ``MouthSmile``, and emotion tags trigger model hotkeys.
The first connection asks you to click "Allow" inside VTube Studio; the token is saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiohttp

from ..config import AvatarConfig

log = logging.getLogger(__name__)

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"


class VTSError(RuntimeError):
    pass


class VTubeStudio:
    def __init__(self, cfg: AvatarConfig, hotkeys: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.emotion_hotkeys = {**(hotkeys or {}), **(cfg.hotkeys or {})}
        self.connected = False
        self.authenticated = False
        self.hotkeys: dict[str, dict] = {}  # lowercase name or id -> hotkey info
        self._mouth = 0.0
        self._smile = 0.5
        self._dirty = True
        self._pending: dict[str, asyncio.Future] = {}
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._tasks: list[asyncio.Task] = []
        self._active_toggle: Optional[str] = None
        self._revert: Optional[asyncio.TimerHandle] = None
        self._closing = False

    # ------------------------------------------------------------------ public controls
    def set_mouth(self, level: float) -> None:
        value = max(0.0, min(1.0, level * self.cfg.lipsync_gain))
        if abs(value - self._mouth) > 0.01:
            self._mouth = value
            self._dirty = True

    def set_smile(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if abs(value - self._smile) > 0.02:
            self._smile = value
            self._dirty = True

    async def express(self, emotion: str) -> bool:
        """Trigger the hotkey mapped to an emotion. Toggle expressions revert after a hold time."""
        ref = self.emotion_hotkeys.get(emotion)
        if not ref or not self.authenticated:
            return False
        info = self.hotkeys.get(str(ref).lower())
        hotkey_id = info["hotkeyID"] if info else ref
        is_toggle = bool(info and info.get("type") == "ToggleExpression")
        try:
            if is_toggle and self._active_toggle and self._active_toggle != hotkey_id:
                await self.request("HotkeyTriggerRequest", {"hotkeyID": self._active_toggle})
                self._active_toggle = None
            if is_toggle and self._active_toggle == hotkey_id:
                self._schedule_revert(hotkey_id)
                return True
            await self.request("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})
        except (VTSError, asyncio.TimeoutError) as exc:
            log.debug("VTS hotkey %s failed: %s", ref, exc)
            return False
        if is_toggle:
            self._active_toggle = hotkey_id
            self._schedule_revert(hotkey_id)
        return True

    def _schedule_revert(self, hotkey_id: str) -> None:
        if self._revert:
            self._revert.cancel()
        loop = asyncio.get_running_loop()

        def fire() -> None:
            if self._active_toggle == hotkey_id and self.authenticated:
                self._active_toggle = None
                asyncio.ensure_future(self._safe_request("HotkeyTriggerRequest", {"hotkeyID": hotkey_id}))

        self._revert = loop.call_later(self.cfg.expression_hold_s, fire)

    async def _safe_request(self, msg_type: str, data: dict) -> None:
        try:
            await self.request(msg_type, data)
        except Exception as exc:  # noqa: BLE001
            log.debug("VTS %s failed: %s", msg_type, exc)

    # ------------------------------------------------------------------ protocol
    async def request(self, msg_type: str, data: Optional[dict] = None, timeout: float = 10.0) -> dict:
        if self._ws is None or self._ws.closed:
            raise VTSError("not connected to VTube Studio")
        rid = uuid.uuid4().hex[:12]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send_str(json.dumps({
            "apiName": API_NAME, "apiVersion": API_VERSION, "requestID": rid,
            "messageType": msg_type, "data": data or {},
        }))
        try:
            reply = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)
        if reply.get("messageType") == "APIError":
            err = reply.get("data", {})
            raise VTSError(f"{err.get('errorID')}: {err.get('message')}")
        return reply.get("data", {})

    async def _reader(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            fut = self._pending.get(payload.get("requestID", ""))
            if fut is not None and not fut.done():
                fut.set_result(payload)

    async def _authenticate(self) -> None:
        token_path = Path(self.cfg.token_path)
        token = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
        ident = {"pluginName": self.cfg.plugin_name, "pluginDeveloper": self.cfg.plugin_developer}
        if token:
            result = await self.request("AuthenticationRequest", {**ident, "authenticationToken": token})
            if result.get("authenticated"):
                self.authenticated = True
                return
        log.warning("VTube Studio: click 'Allow' in the VTube Studio window to authorise %s", self.cfg.plugin_name)
        result = await self.request("AuthenticationTokenRequest", ident, timeout=120)
        token = result.get("authenticationToken", "")
        if not token:
            raise VTSError("VTube Studio did not issue a token")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token, encoding="utf-8")
        result = await self.request("AuthenticationRequest", {**ident, "authenticationToken": token})
        self.authenticated = bool(result.get("authenticated"))
        if not self.authenticated:
            raise VTSError(f"authentication failed: {result.get('reason')}")

    async def refresh_hotkeys(self) -> list[dict]:
        result = await self.request("HotkeysInCurrentModelRequest", {})
        hotkeys = result.get("availableHotkeys", [])
        self.hotkeys = {}
        for hk in hotkeys:
            self.hotkeys[str(hk.get("hotkeyID", "")).lower()] = hk
            if hk.get("name"):
                self.hotkeys[str(hk["name"]).lower()] = hk
        return hotkeys

    async def _inject_loop(self) -> None:
        last_send = 0.0
        while self.authenticated and self._ws is not None and not self._ws.closed:
            now = time.monotonic()
            if self._dirty or now - last_send > 0.5 or self._mouth > 0:
                self._dirty = False
                last_send = now
                values = [{"id": self.cfg.mouth_param, "value": round(self._mouth, 3)}]
                if self.cfg.smile_param:
                    values.append({"id": self.cfg.smile_param, "value": round(self._smile, 3)})
                try:
                    # Fire-and-forget: 30 round trips a second would only add latency.
                    await self._ws.send_str(json.dumps({
                        "apiName": API_NAME, "apiVersion": API_VERSION, "requestID": "inject",
                        "messageType": "InjectParameterDataRequest",
                        "data": {"faceFound": False, "mode": "set", "parameterValues": values},
                    }))
                except (ConnectionError, RuntimeError) as exc:
                    log.debug("parameter injection failed: %s", exc)
                    return
            await asyncio.sleep(1 / 30)

    async def _run(self) -> None:
        delay = 2.0
        while not self._closing:
            helpers: list[asyncio.Task] = []
            try:
                self._session = self._session or aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(self.cfg.url, heartbeat=20)
                self.connected = True
                reader = asyncio.ensure_future(self._reader())
                helpers.append(reader)
                await self._authenticate()
                hotkeys = await self.refresh_hotkeys()
                log.info("VTube Studio connected (%d hotkeys)", len(hotkeys))
                delay = 2.0
                helpers.append(asyncio.ensure_future(self._inject_loop()))
                await reader
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep retrying; VTS may start later
                log.info("VTube Studio unavailable at %s (%s); retrying in %.0fs", self.cfg.url, exc, delay)
            finally:
                self.connected = self.authenticated = False
                for task in helpers:
                    task.cancel()
                if self._ws is not None and not self._ws.closed:
                    await self._ws.close()
            if self._closing:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def start(self) -> None:
        self._tasks.append(asyncio.ensure_future(self._run()))

    async def aclose(self) -> None:
        self._closing = True
        if self._revert:
            self._revert.cancel()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

    def status(self) -> dict[str, Any]:
        return {"connected": self.connected, "authenticated": self.authenticated,
                "hotkeys": len({h.get("hotkeyID") for h in self.hotkeys.values()})}
