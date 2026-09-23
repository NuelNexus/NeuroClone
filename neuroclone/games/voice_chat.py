"""The Neuro SDK voice chat side-channel (API/VOICE_CHAT.md).

Games open ``/game/<name>/voice`` next to their main connection. Binary frames upstream are
``[u8 version=1][u8 flags][u16 speaker id LE][float32 PCM @48 kHz mono]``; we transcribe each
speaker separately so the character hears who said what. Downstream we send the character's
voice as headerless 48 kHz float32 PCM, keyed with ``voice/speaking`` and ``voice/cancelled``.
Without a speech recogniser we answer ``voice/unavailable`` and the game continues without voice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from aiohttp import WSMsgType, web

from ..events import EventBus, VoiceTranscript
from ..speech.audio import AudioClip
from ..speech.stt import Transcriber

log = logging.getLogger(__name__)

HEADER = struct.Struct("<BBH")
SAMPLE_RATE = 48000
MAX_SPEAKERS = 32
MAX_BUFFER_S = 30


@dataclass
class _VoiceClient:
    game: str
    ws: web.WebSocketResponse
    speakers: dict = field(default_factory=dict)  # id -> name
    buffers: dict = field(default_factory=dict)  # id -> list[np.ndarray]
    last_audio: dict = field(default_factory=dict)  # id -> monotonic ts
    active: bool = False


class VoiceChatHub:
    def __init__(self, transcriber: Optional[Transcriber], submit: Callable[[object], None],
                 bus: Optional[EventBus] = None, silence_s: float = 0.8) -> None:
        self.transcriber = transcriber
        self.submit = submit
        self.bus = bus or EventBus()
        self.silence_s = silence_s
        self.clients: dict[str, _VoiceClient] = {}
        self._send_tasks: set[asyncio.Task] = set()

    async def _send_json(self, ws: web.WebSocketResponse, command: str, data: Optional[dict] = None) -> None:
        msg = {"command": command}
        if data is not None:
            msg["data"] = data
        try:
            await ws.send_str(json.dumps(msg))
        except (ConnectionError, RuntimeError):
            pass

    async def handle(self, request: web.Request, game: str) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        client = _VoiceClient(game, ws)
        flusher = asyncio.ensure_future(self._flush_loop(client))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and not await self._control(client, payload):
                        break
                elif msg.type == WSMsgType.BINARY and client.active:
                    self._audio(client, msg.data)
        finally:
            flusher.cancel()
            if self.clients.get(game) is client:
                del self.clients[game]
        return ws

    async def _control(self, client: _VoiceClient, payload: dict) -> bool:
        command = payload.get("command")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if command == "voice/start":
            if self.transcriber is None:
                await self._send_json(client.ws, "voice/unavailable", {"reason": "speech recognition is not enabled"})
                await client.ws.close()
                return False
            client.active = True
            self.clients[client.game] = client
            await self._send_json(client.ws, "voice/ready", {"sample_rate": SAMPLE_RATE, "channels": 1})
            self.bus.publish("game", {"game": client.game, "event": "voice_ready"})
        elif command == "voice/speakers/register":
            for spk in data.get("speakers") or []:
                try:
                    sid, name = int(spk["id"]), str(spk["name"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= sid <= 0xFFFF and (sid in client.speakers or len(client.speakers) < MAX_SPEAKERS):
                    client.speakers[sid] = name
        elif command == "voice/speakers/unregister":
            for sid in data.get("ids") or []:
                client.speakers.pop(sid, None)
                client.buffers.pop(sid, None)
                client.last_audio.pop(sid, None)
        elif command == "voice/stop":
            await self._flush(client, force=True)
            client.active = False
        return True

    def _audio(self, client: _VoiceClient, frame: bytes) -> None:
        if len(frame) < HEADER.size + 4:
            return
        version, _flags, sid = HEADER.unpack_from(frame)
        if version != 1 or sid not in client.speakers:
            return
        payload = frame[HEADER.size:]
        samples = np.frombuffer(payload[: len(payload) - len(payload) % 4], dtype="<f4")
        chunks = client.buffers.setdefault(sid, [])
        if sum(len(c) for c in chunks) < MAX_BUFFER_S * SAMPLE_RATE:
            chunks.append(samples.copy())
        client.last_audio[sid] = time.monotonic()

    async def _flush_loop(self, client: _VoiceClient) -> None:
        while True:
            await asyncio.sleep(0.1)
            await self._flush(client)

    async def _flush(self, client: _VoiceClient, force: bool = False) -> None:
        now = time.monotonic()
        for sid, last in list(client.last_audio.items()):
            if not force and now - last < self.silence_s:
                continue
            chunks = client.buffers.pop(sid, [])
            client.last_audio.pop(sid, None)
            if not chunks or self.transcriber is None:
                continue
            audio = np.concatenate(chunks)
            try:
                text = await self.transcriber.transcribe(audio, SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                log.warning("voice chat transcription failed: %s", exc)
                continue
            if text:
                speaker = client.speakers.get(sid, f"player {sid}")
                self.submit(VoiceTranscript(speaker=speaker, text=text, source="game_vc"))

    # ------------------------------------------------------------------ downstream (her voice)
    async def speaking(self, is_speaking: bool) -> None:
        for client in list(self.clients.values()):
            await self._send_json(client.ws, "voice/speaking", {"speaking": is_speaking})

    def send_clip(self, clip: AudioClip) -> None:
        if not self.clients:
            return
        task = asyncio.ensure_future(self._stream_clip(clip.resample(SAMPLE_RATE).samples.astype("<f4")))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    async def _stream_clip(self, samples: np.ndarray, chunk_s: float = 0.1) -> None:
        step = int(SAMPLE_RATE * chunk_s)
        for start in range(0, len(samples), step):
            data = samples[start : start + step].tobytes()
            for client in list(self.clients.values()):
                try:
                    await client.ws.send_bytes(data)
                except (ConnectionError, RuntimeError):
                    pass
            await asyncio.sleep(chunk_s * 0.95)  # slightly ahead of real time; the game buffers

    async def cancel(self) -> None:
        for task in list(self._send_tasks):
            task.cancel()
        for client in list(self.clients.values()):
            await self._send_json(client.ws, "voice/cancelled")
