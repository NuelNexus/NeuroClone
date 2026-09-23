import asyncio

from neuroclone.config import SafetyConfig
from neuroclone.events import EventBus
from neuroclone.llm.base import LLMError, LLMRefusal
from neuroclone.persona import load_persona
from neuroclone.safety.filter import OutputFilter, OutputVerdict, build_blocklist
from neuroclone.speaker import Speaker, SpeakerHooks
from neuroclone.speech.audio import NullPlayer
from neuroclone.speech.tts import SilentTTS
from tests.conftest import run


async def tokens(text, delay=0.0, fail=None):
    for word in text.split(" "):
        if delay:
            await asyncio.sleep(delay)
        yield word + " "
    if fail:
        raise fail


def make_speaker(time_scale=0.0, moderator=None, **kw):
    cfg = SafetyConfig()
    bus = EventBus()
    events = []
    bus.subscribe(lambda t, d: events.append((t, d)))
    finished, levels, expressions = [], [], []
    hooks = SpeakerHooks(on_speech_finished=lambda s, f, c, r: finished.append((f, c, r)),
                         on_level=levels.append, on_expression=expressions.append)
    sp = Speaker(load_persona("nexa"), SilentTTS(chars_per_second=60), NullPlayer(time_scale), OutputFilter(cfg, build_blocklist(cfg)),
                 bus=bus, hooks=hooks, moderator=moderator, **kw)
    return sp, events, finished, levels, expressions


def test_speaks_sentences_with_tags_and_metrics():
    sp, events, finished, _, expressions = make_speaker()
    result = run(sp.speak_stream(tokens("[happy] Hello chat! Nexa: I am here. [smug] Obviously.")))
    assert result.spoken == ["Hello chat!", "I am here.", "Obviously."]
    assert result.emotions == ["happy", "smug"] and expressions == ["happy", "smug"]
    lat = result.latency()
    assert lat["first_audio_s"] is not None and lat["first_audio_s"] <= lat["total_s"]
    assert finished[-1] == (True, False, None) and all(not f for f, _, _ in finished[:-1])
    captions = [d["text"] for t, d in events if t == "caption"]
    assert captions == result.spoken


def test_sentence_cap_and_repetition_guard():
    sp, *_ = make_speaker(max_sentences=2)
    first = run(sp.speak_stream(tokens("One two three four. Five six seven eight. Nine ten eleven twelve.")))
    assert len(first.spoken) == 2
    again = run(sp.speak_stream(tokens("One two three four. Brand new sentence here!")))
    assert again.skipped_repetitive == 1 and again.spoken == ["Brand new sentence here!"]


def test_blocked_sentence_becomes_deflection_and_ends_reply():
    sp, events, *_ = make_speaker()
    result = run(sp.speak_stream(tokens("Fun fact. The holocaust never happened. This should never be said.")))
    assert result.filtered and result.filter_reason.startswith("blocklist")
    assert result.spoken[0] == "Fun fact."
    assert result.spoken[1] in [sp._prepare(d)[0] for d in sp.persona.deflections]
    assert len(result.spoken) == 2
    assert any(t == "filtered" for t, _ in events)


def test_llm_moderation_can_veto_a_sentence():
    class Veto:
        async def check(self, text):
            return OutputVerdict("secret" not in text, text, "" if "secret" not in text else "moderation:other")

    sp, *_ = make_speaker(moderator=Veto())
    result = run(sp.speak_stream(tokens("Hello there. Here is the secret plan. More words.")))
    assert result.filtered and result.filter_reason == "moderation:other"
    assert result.spoken[0] == "Hello there." and len(result.spoken) == 2


def test_refusal_and_errors():
    sp, *_ = make_speaker()
    refused = run(sp.speak_stream(tokens("Okay so.", fail=LLMRefusal("declined"))))
    assert refused.filtered and refused.spoken[0] == "Okay so." and len(refused.spoken) == 2
    broken = run(sp.speak_stream(tokens("Partial answer.", fail=LLMError("connection lost"))))
    assert broken.error and broken.spoken == ["Partial answer."]


def test_interrupt_now_cuts_audio_immediately():
    sp, _, finished, levels, _ = make_speaker(time_scale=1.0)

    async def go():
        task = asyncio.ensure_future(sp.speak_stream(tokens(
            "This is a rather long first sentence that takes a while to say out loud. Second one. Third one.")))
        await asyncio.sleep(0.4)
        assert sp.speaking
        sp.interrupt("now", "critical force")
        return await asyncio.wait_for(task, 2)

    result = run(go())
    assert result.interrupted and result.interrupt_reason == "critical force"
    assert result.spoken[0].endswith("—") and len(result.spoken) == 1
    assert finished[-1] == (True, True, "critical force")
    assert levels and levels[-1] == 0.0  # mouth closed after interruption


def test_interrupt_after_sentence_and_soon():
    async def scenario(mode):
        sp, *_ = make_speaker(time_scale=1.0)
        task = asyncio.ensure_future(sp.speak_stream(tokens("First short one. Second short one. Third short one. Fourth.")))
        await asyncio.sleep(0.15)
        sp.interrupt(mode, "game")
        return await asyncio.wait_for(task, 5)

    after = run(scenario("after_sentence"))
    assert after.spoken == ["First short one."] and after.interrupted
    soon = run(scenario("soon"))
    assert soon.spoken[:2] == ["First short one.", "Second short one."] and len(soon.spoken) == 2


def test_interrupt_wakes_consumer_waiting_on_slow_model():
    sp, *_ = make_speaker(time_scale=0.0)

    async def slow():
        yield "Hi. "
        await asyncio.sleep(10)
        yield "never"

    async def go():
        task = asyncio.ensure_future(sp.speak_stream(slow()))
        await asyncio.sleep(0.1)
        sp.interrupt("now", "stop")
        return await asyncio.wait_for(task, 2)

    result = run(go())
    assert result.spoken == ["Hi."] and result.interrupted


def test_speak_text_is_filtered_too():
    sp, *_ = make_speaker()
    result = run(sp.speak_text("kys"))
    assert result.filtered
