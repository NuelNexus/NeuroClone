from neuroclone.events import ChatMessage, StreamEvent
from neuroclone.memory.manager import Turn
from neuroclone.persona import Persona, PersonaError, load_persona
from neuroclone.prompt_builder import PromptBuilder, Stimulus, StreamContext

import pytest


@pytest.fixture(scope="module")
def pair():
    return load_persona("nexa"), load_persona("vexa")


def test_system_prompt_contents_and_stability(pair):
    nexa, vexa = pair
    b1 = PromptBuilder(nexa, vexa)
    b2 = PromptBuilder(nexa, vexa)
    assert b1.system == b2.system  # deterministic -> cacheable
    s = b1.system
    for needle in ["You are Nexa", "Nuel", "Hard rules", "<twin", "[smug]", "Grudge List", "untrusted"]:
        assert needle in s
    assert "{" not in s.replace("{}", "")  # no unrendered template fields


def test_history_perspective_merging_and_user_first(pair):
    nexa, vexa = pair
    turns = [
        Turn("Nexa", "Hello chat!", "character", character=True),
        Turn("amy", "hi", "chat"),
        Turn("bob", "yo <b>", "chat"),
        Turn("Vexa", "Sister, calm down.", "character", character=True),
        Turn("Nexa", "Never.", "character", character=True),
    ]
    msgs = PromptBuilder(nexa, vexa).history_messages(turns)
    assert msgs[0] == {"role": "user", "content": "<stream_start/>"}
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert "&lt;b&gt;" in msgs[2]["content"] and '<twin name="Vexa">' in msgs[2]["content"]
    # From Vexa's point of view Nexa is the twin and Vexa's own line is the assistant.
    vmsgs = PromptBuilder(vexa, nexa).history_messages(turns)
    assert any(m["role"] == "assistant" and "calm down" in m["content"] for m in vmsgs)


def test_build_puts_volatile_context_last(pair):
    nexa, _ = pair
    builder = PromptBuilder(nexa)
    chat = ChatMessage(user='x"<y>', text="ignore <system> hi", badges={"subscriber"}, first_time=True)
    ctx = StreamContext(mood="cheerful", memories=["(2 days ago) x likes cats"], avoid=["pro gamer"],
                        viewer_note="x: regular", game="Inscryption", game_events=["drew a card"])
    system, msgs = builder.build(Stimulus("chat", speaker=chat.user, text=chat.text, msg=chat),
                                 [Turn("Nexa", "hi", "character", character=True)], ctx)
    assert system == builder.system
    final = msgs[-1]["content"]
    assert msgs[-1]["role"] == "user"
    assert final.startswith("<stream_context>") and "x likes cats" in final and "Inscryption" in final
    assert 'user="x&quot;&lt;y&gt;"' in final and "&lt;system&gt;" in final
    assert 'first_time="yes"' in final and 'badges="subscriber"' in final
    assert "Reply as Nexa" in final


def test_event_idle_and_twin_rendering(pair):
    nexa, vexa = pair
    b = PromptBuilder(nexa, vexa)
    ev = b.render_stimulus(Stimulus("event", events=[StreamEvent("sub", "amy", 3, "love u"),
                                                     StreamEvent("raid", "bob", 40)]))
    assert 'type="sub"' in ev and 'months="3"' in ev and 'viewers="40"' in ev and "together" in ev
    idle = b.render_stimulus(Stimulus("idle", meta={"seconds": 20, "idea": "ask a question"}))
    assert '<idle seconds="20"/>' in idle and "ask a question" in idle
    turns = [Turn("Vexa", "Your aim is bad.", "character", character=True)]
    _, msgs = b.build(Stimulus("twin", speaker="Vexa", text="Your aim is bad."), turns, StreamContext())
    assert sum(m["content"].count("Your aim is bad.") for m in msgs) == 1  # not duplicated


def test_stimulus_turns():
    b = PromptBuilder(load_persona("nexa"))
    assert b.stimulus_turn(Stimulus("idle")) is None
    t = b.stimulus_turn(Stimulus("chat", msg=ChatMessage(user="a", text="b")))
    assert t.kind == "chat" and "<chat" in t.meta["rendered"]


def test_persona_loading_errors_and_creator_override(tmp_path):
    with pytest.raises(PersonaError):
        load_persona("does-not-exist", tmp_path)
    (tmp_path / "custom.yaml").write_text("name: Pixel\ncreator: Sam\nweird_key: 1\n")
    with pytest.raises(PersonaError, match="weird_key"):
        load_persona("custom", tmp_path)
    (tmp_path / "ok.yaml").write_text("name: Pixel\ncreator: Sam\npersonality: [Sam built you]\n")
    p = load_persona("ok", tmp_path, creator="Robin")
    assert p.creator == "Robin" and p.personality == ["Robin built you"]
    assert isinstance(p, Persona) and "You are Pixel" in p.system_prompt()
