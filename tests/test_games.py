import asyncio
import json
import struct

import aiohttp
import numpy as np

from neuroclone.config import GamesConfig
from neuroclone.events import ActionForce, EventBus, GameContext, VoiceTranscript
from neuroclone.games.agent import GameAgent, decision_schema
from neuroclone.games.neuro_api import NeuroApiServer
from neuroclone.games.voice_chat import VoiceChatHub
from neuroclone.llm.base import LLM
from neuroclone.speech.audio import AudioClip
from neuroclone.speech.stt import Transcriber
from tests.conftest import run

ACTIONS = [
    {"name": "play_card", "description": "Play a card from your hand.",
     "schema": {"type": "object", "properties": {"card": {"type": "string", "enum": ["ace", "king"]},
                                                 "lane": {"type": "integer", "minimum": 1, "maximum": 4}},
                "required": ["card", "lane"]}},
    {"name": "end_turn", "description": "End your turn."},
]


class Harness:
    def __init__(self, **cfg):
        self.events = []
        self.server = NeuroApiServer(GamesConfig(port=0, host="127.0.0.1", **cfg), "nexa", "Nexa",
                                     self.events.append, EventBus())

    async def __aenter__(self):
        await self.server.start()
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        await self.server.stop()

    async def connect(self, path=""):
        return await self.session.ws_connect(f"ws://127.0.0.1:{self.server.port}/{path}")

    async def wait_for(self, predicate, timeout=3.0):
        for _ in range(int(timeout / 0.02)):
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return False


async def send(ws, command, data=None, game="Card Game"):
    msg = {"command": command, "game": game}
    if data is not None:
        msg["data"] = data
    await ws.send_str(json.dumps(msg))


def test_startup_ack_register_context_force_and_results():
    async def go():
        async with Harness() as h:
            ws = await h.connect()
            await send(ws, "startup")
            ack = json.loads((await ws.receive()).data)
            assert ack == {"command": "startup", "data": {"session": {
                "sessionId": h.server.sessions["Card Game"].session_id, "characterId": "nexa", "displayName": "Nexa"}}}
            await send(ws, "actions/register", {"actions": ACTIONS})
            await send(ws, "context", {"message": "You drew an ace.", "silent": False})
            await send(ws, "definitely/unknown", {"x": 1})  # ignored
            await ws.send_str("{not json")  # ignored
            await send(ws, "actions/force", {"state": "## Hand\n- ace", "query": "Your turn.",
                                             "action_names": ["play_card", "ghost"], "priority": "high"})
            assert await h.wait_for(lambda: any(isinstance(e, ActionForce) for e in h.events))
            ctx = next(e for e in h.events if isinstance(e, GameContext))
            assert ctx.message == "You drew an ace." and ctx.silent is False
            force = next(e for e in h.events if isinstance(e, ActionForce))
            assert force.action_names == ["play_card"] and force.priority == "high" and force.state.startswith("## Hand")
            assert h.server.pending_force("Card Game") is force
            assert sorted(a.name for a in h.server.actions("Card Game")) == ["end_turn", "play_card"]

            async def game_side():
                msg = json.loads((await ws.receive()).data)
                assert msg["command"] == "action" and msg["data"]["name"] == "play_card"
                assert json.loads(msg["data"]["data"]) == {"card": "ace", "lane": 2}
                await send(ws, "action/result", {"id": msg["data"]["id"], "success": True, "message": "Played."})

            result, _ = await asyncio.gather(
                h.server.execute("Card Game", "play_card", json.dumps({"card": "ace", "lane": 2})), game_side())
            assert result.success and result.message == "Played."

            await h.server.speech_finished(True, cancelled=True, reason="interrupted")
            msg = json.loads((await ws.receive()).data)
            assert msg == {"command": "speech_finished", "data": {"isFinal": True, "cancelled": True, "reason": "interrupted"}}
            await ws.close()

    run(go())


def test_timeout_unregister_and_context_survives_reconnect():
    async def go():
        async with Harness(action_timeout_s=0.3) as h:
            ws = await h.connect()
            await send(ws, "startup")
            await ws.receive()
            await send(ws, "actions/register", {"actions": ACTIONS})
            await send(ws, "context", {"message": "Round 1 started.", "silent": True})
            await send(ws, "actions/force", {"query": "Go", "action_names": ["play_card"]})
            assert await h.wait_for(lambda: h.server.pending_force("Card Game") is not None)
            result = await h.server.execute("Card Game", "end_turn")
            assert not result.success and result.timed_out  # game never answered
            await ws.receive()  # the action message we ignored
            await send(ws, "actions/unregister", {"action_names": ["play_card"]})
            assert await h.wait_for(lambda: h.server.pending_force("Card Game") is None)
            await ws.close()
            assert await h.wait_for(lambda: not h.server.sessions["Card Game"].connected)
            ws2 = await h.connect("game/Card%20Game")
            await send(ws2, "startup")
            await ws2.receive()
            assert "Round 1 started." in h.server.context_log("Card Game")  # context survives
            assert h.server.actions("Card Game") == []  # startup clears actions
            await ws2.close()
            status = await (await h.session.get(f"http://127.0.0.1:{h.server.port}/")).json()
            assert status["service"] == "neuroclone-neuro-api"

    run(go())


class FakeTranscriber(Transcriber):
    def __init__(self):
        self.calls = []

    async def transcribe(self, samples, sample_rate=16000):
        self.calls.append((len(samples), sample_rate))
        return "hello from the lobby"


def test_voice_chat_side_channel():
    async def go():
        async with Harness() as h:
            stt = FakeTranscriber()
            h.server.voice = VoiceChatHub(stt, h.events.append, silence_s=0.2)
            vws = await h.connect("game/Card%20Game/voice")
            await vws.send_str(json.dumps({"command": "voice/start", "game": "Card Game"}))
            ready = json.loads((await vws.receive()).data)
            assert ready == {"command": "voice/ready", "data": {"sample_rate": 48000, "channels": 1}}
            await vws.send_str(json.dumps({"command": "voice/speakers/register", "game": "Card Game",
                                           "data": {"speakers": [{"id": 7, "name": "Vedal"}]}}))
            pcm = (np.sin(np.linspace(0, 200, 9600)) * 0.3).astype("<f4").tobytes()
            await vws.send_bytes(struct.pack("<BBH", 1, 0, 7) + pcm)
            await vws.send_bytes(struct.pack("<BBH", 1, 0, 99) + pcm)  # unregistered speaker: dropped
            assert await h.wait_for(lambda: any(isinstance(e, VoiceTranscript) for e in h.events))
            vt = next(e for e in h.events if isinstance(e, VoiceTranscript))
            assert vt.speaker == "Vedal" and vt.source == "game_vc"
            assert stt.calls == [(9600, 48000)]
            # Downstream: her voice + keying messages.
            await h.server.voice.speaking(True)
            assert json.loads((await vws.receive()).data) == {"command": "voice/speaking", "data": {"speaking": True}}
            h.server.voice.send_clip(AudioClip(np.zeros(2400, dtype=np.float32), 24000))
            frame = await vws.receive()
            assert frame.type == aiohttp.WSMsgType.BINARY and len(frame.data) == 4800 * 4
            await h.server.voice.cancel()
            assert json.loads((await vws.receive()).data)["command"] == "voice/cancelled"
            await vws.close()

            # Without a transcriber the channel politely declines.
            h.server.voice = VoiceChatHub(None, h.events.append)
            vws2 = await h.connect("game/Card%20Game/voice")
            await vws2.send_str(json.dumps({"command": "voice/start", "game": "Card Game"}))
            assert json.loads((await vws2.receive()).data)["command"] == "voice/unavailable"

    run(go())


class ScriptedLLM(LLM):
    name = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    async def stream(self, system, messages, *, max_tokens=None, purpose="chat"):
        yield "ok"

    async def complete_json(self, system, messages, schema, *, name="output", max_tokens=None):
        self.prompts.append(messages[-1]["content"])
        return self.answers.pop(0)


def agent_with(answers, **cfg):
    server = NeuroApiServer(GamesConfig(**cfg), "nexa", "Nexa", lambda e: None)
    server.add_context("Card Game", "Opponent played a king.")
    from neuroclone.games.neuro_api import ActionDef

    session = server.sessions["Card Game"]
    for a in ACTIONS:
        session.actions[a["name"]] = ActionDef(a["name"], a["description"], a.get("schema"))
    llm = ScriptedLLM(answers)
    return GameAgent(llm, server, "You are Nexa.", server.cfg), llm


def test_agent_validates_and_retries_with_feedback():
    agent, llm = agent_with([
        {"say": "hmm", "action": "play_card", "data": '{"card": "queen", "lane": 9}'},
        {"say": "Take this!", "action": "play_card", "data": '{"card": "ace", "lane": 3}'},
    ])
    force = ActionForce(game="Card Game", query="Your turn.", action_names=["play_card", "end_turn"], state="Hand: ace")
    decision = run(agent.decide("Card Game", force))
    assert decision.action == "play_card" and json.loads(decision.data) == {"card": "ace", "lane": 3}
    assert decision.say == "Take this!" and decision.attempts == 2
    assert "does not match the schema" in llm.prompts[1]
    assert "Opponent played a king." in llm.prompts[0] and "Hand: ace" in llm.prompts[0]


def test_agent_falls_back_to_a_valid_random_action_and_handles_none():
    agent, _ = agent_with([{"say": "", "action": "fly", "data": "{}"}, {"say": "", "action": "play_card", "data": "nope"}])
    force = ActionForce(game="Card Game", query="Go", action_names=["play_card"])
    decision = run(agent.decide("Card Game", force))
    assert decision.fallback and decision.action == "play_card"
    assert json.loads(decision.data)["card"] in ("ace", "king")

    agent, _ = agent_with([{"say": "Not yet.", "action": "none", "data": "{}"}])
    decision = run(agent.decide("Card Game", None, allow_none=True))
    assert decision.action is None and decision.say == "Not yet."

    agent, _ = agent_with([{"say": "", "action": "end_turn", "data": "{}"}])
    decision = run(agent.decide("Card Game", ActionForce(game="Card Game", query="q", action_names=["end_turn"])))
    assert decision.action == "end_turn" and decision.data is None  # schema-less actions send no data


def test_decision_schema_is_strict_compatible():
    schema = decision_schema(["a", "b"], allow_none=True)
    assert schema["additionalProperties"] is False and set(schema["required"]) == {"say", "action", "data"}
    assert schema["properties"]["action"]["enum"] == ["a", "b", "none"]
