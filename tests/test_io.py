"""Chat sources, VTube Studio, audio and TTS."""

import asyncio
import io
import json
import struct
import wave

import numpy as np
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from neuroclone.avatar.vtube_studio import VTubeStudio
from neuroclone.chat.console import parse_console_line
from neuroclone.chat.twitch import TwitchChat, parse_line, to_events
from neuroclone.chat.youtube import item_to_events
from neuroclone.config import AvatarConfig, TTSConfig, TwitchConfig
from neuroclone.emotion import VoiceStyle
from neuroclone.events import ChatMessage, GameContext, ModeratorCommand, StreamEvent, VoiceTranscript
from neuroclone.persona import load_persona
from neuroclone.speech.audio import AudioClip, NullPlayer, WavDumpPlayer
from neuroclone.speech.tts import AzureTTS, OpenAITTS, SilentTTS, TTSError, create_tts
from tests.conftest import run


# ---------------------------------------------------------------- Twitch
def test_twitch_privmsg_with_tags():
    line = ("@badge-info=subscriber/5;badges=subscriber/3,premium/1;bits=100;display-name=Cool\\sGuy;"
            "first-msg=1;user-id=42 :coolguy!coolguy@coolguy.tmi.twitch.tv PRIVMSG #nexa :Cheer100 hi nexa!")
    events = to_events(parse_line(line))
    chat, bits = events
    assert isinstance(chat, ChatMessage) and chat.user == "Cool Guy" and chat.text == "Cheer100 hi nexa!"
    assert chat.badges == {"subscriber", "premium"} and chat.is_sub and chat.first_time and chat.bits == 100
    assert isinstance(bits, StreamEvent) and bits.kind == "bits" and bits.amount == 100


@pytest.mark.parametrize("tags,kind,amount,user", [
    ("msg-id=resub;msg-param-cumulative-months=7;display-name=Amy", "resub", 7, "Amy"),
    ("msg-id=submysterygift;msg-param-mass-gift-count=5;display-name=Bob", "gift", 5, "Bob"),
    ("msg-id=raid;msg-param-viewerCount=321;msg-param-displayName=Raider;display-name=raider", "raid", 321, "Raider"),
])
def test_twitch_usernotices(tags, kind, amount, user):
    ev = to_events(parse_line(f"@{tags} :tmi.twitch.tv USERNOTICE #nexa :great stream"))[0]
    assert ev.kind == kind and ev.amount == amount and ev.user == user


def test_twitch_client_against_fake_irc_server():
    received, sent = [], []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        for _ in range(3):  # NICK, CAP, JOIN
            sent.append((await ws.receive()).data)
        await ws.send_str("PING :tmi.twitch.tv\r\n"
                          "@display-name=Amy;badges=moderator/1 :amy!amy@amy.tmi.twitch.tv PRIVMSG #chan :hello!\r\n")
        pong = await ws.receive()
        sent.append(pong.data)
        await asyncio.sleep(0.2)
        await ws.close()
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/", handler)
        server = TestServer(app)
        await server.start_server()
        client = TwitchChat(TwitchConfig(channel="#Chan"), received.append, url=f"ws://127.0.0.1:{server.port}/")
        task = asyncio.ensure_future(client.run())
        for _ in range(100):
            if received and len(sent) >= 4:
                break
            await asyncio.sleep(0.02)
        await client.aclose()
        task.cancel()
        await server.close()

    run(go())
    assert sent[0].startswith("NICK justinfan") and sent[2] == "JOIN #chan" and sent[3] == "PONG :tmi.twitch.tv"
    assert received[0].user == "Amy" and received[0].is_mod and received[0].text == "hello!"


# ---------------------------------------------------------------- YouTube + console
def test_youtube_items():
    chat = item_to_events({"snippet": {"type": "textMessageEvent", "displayMessage": "yo",
                                       "textMessageDetails": {"messageText": "yo nexa"}},
                           "authorDetails": {"displayName": "Kim", "isChatSponsor": True, "channelId": "c1"}})[0]
    assert chat.text == "yo nexa" and chat.badges == {"member"}
    sc = item_to_events({"snippet": {"type": "superChatEvent", "superChatDetails": {
        "amountMicros": "5000000", "amountDisplayString": "$5.00", "userComment": "for snacks"}},
        "authorDetails": {"displayName": "Lee"}})
    assert sc[0].kind == "superchat" and sc[0].amount == 5.0 and sc[0].display_amount == "$5.00"
    assert sc[1].text == "for snacks"
    gift = item_to_events({"snippet": {"type": "membershipGiftingEvent",
                                       "membershipGiftingDetails": {"giftMembershipsCount": 10}},
                           "authorDetails": {"displayName": "Max"}})[0]
    assert gift.kind == "gift" and gift.amount == 10


def test_console_lines():
    assert parse_console_line("alice: hi there").user == "alice"
    assert parse_console_line("just text").user == "viewer"
    voice = parse_console_line("> hello Nexa", creator="Nuel")
    assert isinstance(voice, VoiceTranscript) and voice.speaker == "Nuel"
    assert parse_console_line("/bits bob 500 nice").amount == 500
    assert isinstance(parse_console_line("/game boss appeared"), GameContext)
    assert parse_console_line("/say hi").args == {"text": "hi"}
    assert isinstance(parse_console_line("/bits"), ModeratorCommand)
    assert parse_console_line("   ") is None
    assert parse_console_line("time: 10pm is late").user == "time"


# ---------------------------------------------------------------- VTube Studio
def test_vtube_studio_auth_hotkeys_and_lipsync(tmp_path):
    seen = []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            req = json.loads(msg.data)
            seen.append(req)
            kind, data = req["messageType"], {}
            if kind == "AuthenticationTokenRequest":
                data = {"authenticationToken": "tok123"}
            elif kind == "AuthenticationRequest":
                data = {"authenticated": req["data"]["authenticationToken"] == "tok123"}
            elif kind == "HotkeysInCurrentModelRequest":
                data = {"availableHotkeys": [{"name": "Smile", "type": "ToggleExpression", "hotkeyID": "hk1"}]}
            await ws.send_str(json.dumps({"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
                                          "requestID": req["requestID"], "messageType": kind.replace("Request", "Response"),
                                          "data": data}))
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/", handler)
        server = TestServer(app)
        await server.start_server()
        cfg = AvatarConfig(enabled=True, url=f"ws://127.0.0.1:{server.port}/", token_path=str(tmp_path / "tok.txt"),
                           hotkeys={"happy": "Smile"}, expression_hold_s=0.2)
        vts = VTubeStudio(cfg)
        await vts.start()
        for _ in range(200):
            if vts.authenticated and vts.hotkeys:
                break
            await asyncio.sleep(0.01)
        assert vts.status()["authenticated"] and vts.status()["hotkeys"] == 1
        assert await vts.express("happy")
        assert not await vts.express("sad")  # unmapped emotion
        vts.set_mouth(0.5)
        await asyncio.sleep(0.15)
        await asyncio.sleep(0.3)  # toggle expression reverts after the hold time
        await vts.aclose()
        await server.close()

    run(go())
    assert (tmp_path / "tok.txt").read_text() == "tok123"
    triggers = [r for r in seen if r["messageType"] == "HotkeyTriggerRequest"]
    assert [t["data"]["hotkeyID"] for t in triggers] == ["hk1", "hk1"]  # on, then auto-off
    injections = [r for r in seen if r["messageType"] == "InjectParameterDataRequest"]
    values = [p["value"] for r in injections for p in r["data"]["parameterValues"] if p["id"] == "MouthOpen"]
    assert max(values) == pytest.approx(0.8)  # 0.5 * lipsync gain 1.6


# ---------------------------------------------------------------- audio
def _wav(samples, rate=22050, width=2, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 2:
            w.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())
        else:
            ints = (np.asarray(samples) * 8388607).astype("<i4")
            w.writeframes(b"".join(struct.pack("<i", int(v))[:3] for v in ints))
    return buf.getvalue()


def test_wav_parsing_variants_and_roundtrip():
    tone = np.sin(np.linspace(0, 100, 4410)) * 0.5
    clip = AudioClip.from_wav(_wav(tone))
    assert clip.sample_rate == 22050 and abs(clip.samples.max() - 0.5) < 0.01
    stereo = AudioClip.from_wav(_wav(np.repeat(tone, 2), channels=2))
    assert len(stereo.samples) == len(tone)
    c24 = AudioClip.from_wav(_wav(tone, width=3))
    assert abs(c24.samples.max() - 0.5) < 0.01
    floats = tone.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    riff = b"RIFF" + struct.pack("<I", 36 + len(floats)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
    riff += b"data" + struct.pack("<I", len(floats)) + floats
    fclip = AudioClip.from_wav(riff)
    assert fclip.sample_rate == 16000 and np.allclose(fclip.samples, tone, atol=1e-6)
    back = AudioClip.from_wav(clip.to_wav_bytes())
    assert np.allclose(back.samples, clip.samples, atol=1e-3)
    assert len(clip.resample(44100).samples) == 2 * len(clip.samples)
    with pytest.raises(ValueError):
        AudioClip.from_wav(b"not a wav")


def test_envelope_follows_loudness():
    rate = 16000
    samples = np.concatenate([np.zeros(rate // 2), np.sin(np.arange(rate // 2) * 0.3) * 0.8]).astype(np.float32)
    env = AudioClip(samples, rate).envelope(fps=60)
    assert env[:25].max() == 0.0 and env[-10:].mean() > 0.8


def test_null_and_wav_players(tmp_path):
    clip = AudioClip(np.zeros(1600, dtype=np.float32), 16000)
    levels = []
    assert run(NullPlayer(time_scale=1.0).play(clip, levels.append)) is True
    assert levels[-1] == 0.0

    async def stopped():
        player = NullPlayer(time_scale=1.0)
        task = asyncio.ensure_future(player.play(AudioClip.silence(2.0)))
        await asyncio.sleep(0.05)
        player.stop()
        return await task

    assert run(stopped()) is False
    dump = WavDumpPlayer(tmp_path / "wavs", time_scale=0)
    run(dump.play(clip))
    assert len(list((tmp_path / "wavs").glob("*.wav"))) == 1


# ---------------------------------------------------------------- TTS
def test_silent_tts_timing_and_envelope():
    clip = run(SilentTTS(chars_per_second=10).synthesize("0123456789" * 2))
    assert 2.0 < clip.duration < 2.5 and not clip.samples.any()
    assert clip.envelope().max() > 0.5  # mouth still moves


def test_azure_ssml_combines_base_and_emotion():
    tts = AzureTTS("key", "eastus", "en-US-SaraNeural", pitch="+12%", rate="+6%")
    ssml = tts.ssml("Fish & <chips>", VoiceStyle(rate_pct=4, pitch_pct=-2))
    assert 'name="en-US-SaraNeural"' in ssml and 'xml:lang="en-US"' in ssml
    assert 'pitch="+10%"' in ssml and 'rate="+10%"' in ssml
    assert "Fish &amp; &lt;chips&gt;" in ssml
    with pytest.raises(TTSError):
        AzureTTS("", "eastus", "v")


def test_create_tts_uses_persona_voice_defaults():
    nexa = load_persona("nexa")
    azure = create_tts(TTSConfig(provider="azure", azure_key="k"), nexa)
    assert azure.voice == "en-US-SaraNeural" and azure.base_pitch == 12
    override = create_tts(TTSConfig(provider="azure", azure_key="k", voice="en-US-AriaNeural", pitch="+0%"), nexa)
    assert override.voice == "en-US-AriaNeural" and override.base_pitch == 0
    assert isinstance(create_tts(TTSConfig(provider="silent"), nexa), SilentTTS)
    oai = create_tts(TTSConfig(provider="openai"), nexa)
    assert oai.voice == "af_bella" and oai.speed == pytest.approx(1.08)
    with pytest.raises(TTSError):
        create_tts(TTSConfig(provider="nope"), nexa)


def test_openai_tts_against_fake_server():
    bodies = []

    async def speech(request):
        bodies.append(await request.json())
        return web.Response(body=_wav(np.zeros(2400), rate=24000), content_type="audio/wav")

    async def go():
        app = web.Application()
        app.router.add_post("/v1/audio/speech", speech)
        server = TestServer(app)
        await server.start_server()
        tts = OpenAITTS(f"http://127.0.0.1:{server.port}/v1", "kokoro", "af_bella", speed=1.0)
        clip = await tts.synthesize("hello", VoiceStyle(rate_pct=10))
        await tts.aclose()
        await server.close()
        return clip

    clip = run(go())
    assert clip.sample_rate == 24000 and clip.text == "hello"
    assert bodies[0]["voice"] == "af_bella" and bodies[0]["response_format"] == "wav" and bodies[0]["speed"] == 1.1
