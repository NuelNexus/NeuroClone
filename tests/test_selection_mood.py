import random
import time

from neuroclone.chat.selector import ChatSelector
from neuroclone.config import SelectorConfig
from neuroclone.emotion import EmotionEngine
from neuroclone.events import ChatMessage
from neuroclone.persona import BitTracker, load_persona
from neuroclone.repetition import RepetitionGuard


def msg(user, text, age=0.0, **kw):
    return ChatMessage(user=user, text=text, ts=time.time() - age, **kw)


def greedy():
    return ChatSelector(SelectorConfig(temperature=0), ["Nexa", "Vexa"], rng=random.Random(0))


def test_mentions_questions_and_support_rank_higher():
    sel = greedy()
    now = time.time()
    plain = sel.score(msg("a", "the weather is nice today"), now)
    mention = sel.score(msg("b", "nexa the weather is nice today"), now)
    question = sel.score(msg("c", "what do you think about rain?"), now)
    cheer = sel.score(msg("d", "take my bits you gremlin", bits=500), now)
    assert mention.score > plain.score and question.score > plain.score and cheer.score > plain.score
    assert "mention" in mention.reasons and "support" in cheer.reasons


def test_spam_short_links_and_freshness():
    sel = greedy()
    now = time.time()
    assert sel.score(msg("a", "lol"), now).score < 1.0
    old = sel.score(msg("a", "what is your favourite game?", age=60), now)
    new = sel.score(msg("b", "what is your favourite game?"), now)
    assert new.score > old.score
    for i in range(4):
        sel.add(msg(f"u{i}", "copypasta copypasta copypasta"))
    sel.add(msg("real", "hey nexa how was your day?"))
    chosen = sel.pick(now)
    assert chosen.msg.user == "real"


def test_fairness_cooldown_and_duplicate_removal():
    sel = greedy()
    now = time.time()
    sel.add(msg("alice", "nexa what's your favourite colour?"))
    sel.add(msg("bob", "what's your favourite colour?"))
    first = sel.pick(now)
    assert first.msg.user == "alice"
    # bob asked the same thing, so it was dropped as answered
    assert sel.pending(now) == 0
    sel.add(msg("alice", "nexa one more question?"))
    sel.add(msg("carol", "nexa do you like cats?"))
    assert sel.pick(now + 1).msg.user == "carol"  # alice was just answered


def test_novelty_prefers_new_topics():
    sel = greedy()
    now = time.time()
    sel.mark_answered(msg("x", "do you like minecraft creepers"), now)
    same = sel.score(msg("y", "do you like minecraft creepers a lot"), now)
    fresh = sel.score(msg("z", "what's your opinion on pineapple pizza"), now)
    assert fresh.reasons["novelty"] > same.reasons["novelty"]


def test_vibe_detection():
    sel = ChatSelector(SelectorConfig(vibe_min_messages=5), ["Nexa"])
    now = time.time()
    for i in range(6):
        sel.add(msg(f"u{i}", "W"))
    vibe = sel.vibe(now)
    assert vibe and '"w"' in vibe
    assert sel.vibe(now) is None  # cooldown


def test_softmax_sampling_varies_but_prefers_best():
    sel = ChatSelector(SelectorConfig(temperature=3.0), ["Nexa"], rng=random.Random(1))
    wins = {"best": 0, "ok": 0}
    for _ in range(200):
        sel.buffer.clear()
        sel.answered_users.clear()
        sel.recent_topics.clear()
        sel.add(msg("best", "nexa what do you think about space?"))
        sel.add(msg("ok", "good morning everyone here"))
        wins[sel.pick().msg.user] += 1
    assert wins["best"] > wins["ok"] > 0


def test_emotion_tags_events_and_decay():
    emo = EmotionEngine(half_life_s=10)
    t0 = 1000.0
    emo._t = t0
    emo.apply_tags(["excited"], now=t0)
    hyped = emo.mood(now=t0)
    assert hyped.valence > 0.4 and hyped.emotion == "excited"
    later = emo.mood(now=t0 + 100)
    assert abs(later.valence - emo.base_v) < 0.05
    emo.apply_event("gift", 2.0, now=t0 + 100)
    assert emo.valence > later.valence
    style = emo.voice_style(["sad"])
    assert style.rate_pct < 0 and style.emotion == "sad"
    assert 0 <= emo.smile() <= 1


def test_repetition_guard():
    g = RepetitionGuard(window=10)
    g.record("I am basically a pro gamer, statistically speaking.")
    assert g.is_repetitive("I am basically a pro gamer, statistically.")
    assert not g.is_repetitive("Chat, what snacks should I pretend to eat?")
    g.record("Let's go!")
    assert g.is_repetitive("Let's go!")  # back to back
    for i in range(4):
        g.record(f"world domination is pending funding round {i}")
    assert any("world domination" in p for p in g.overused_phrases())


def test_bit_tracker_budget():
    persona = load_persona("nexa")
    bits = BitTracker(persona)
    now = 5000.0
    for _ in range(3):
        bits.record("You're going on the Grudge List.", now=now)
    assert "grudge list" in bits.exhausted(now=now)
    assert "grudge list" not in bits.exhausted(now=now + 3601)
