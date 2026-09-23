import time

import numpy as np

from neuroclone.config import MemoryConfig
from neuroclone.llm.mock import MockLLM
from neuroclone.memory import HashingEmbedder, MemoryManager, MemoryStore, Turn, estimate_importance, humanize_age
from tests.conftest import run


def make(tmp_path, llm=None, session="s1", **cfg_kw):
    cfg = MemoryConfig(path=str(tmp_path / "m.sqlite3"), **cfg_kw)
    return MemoryManager(cfg, MemoryStore(cfg.path), HashingEmbedder(), llm, character="Nexa", session_id=session)


def test_hashing_embedder_similarity():
    emb = HashingEmbedder()
    a, b, c = run(emb.embed(["my cat Mochi loves tuna", "Mochi the cat eats tuna", "the stock market crashed"]))
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert float(a @ b) > float(a @ c) + 0.2


def test_store_persists_and_reloads_index(tmp_path):
    mm = make(tmp_path)
    run(mm.remember("alice has a dog named Biscuit", kind="fact", subject="alice"))
    mm.store.close()
    reopened = MemoryStore(tmp_path / "m.sqlite3")
    assert reopened.count() == 1
    hits = reopened.search(HashingEmbedder().vector("Biscuit the dog"), k=3)
    assert hits and "Biscuit" in hits[0][0].text
    assert reopened.by_subject("ALICE")[0].subject == "alice"


def test_index_grows_past_initial_capacity(tmp_path):
    mm = make(tmp_path)

    async def fill():
        for i in range(150):
            await mm.remember(f"memory number {i} about topic{i}", kind="episode")

    run(fill())
    assert mm.store.count() == 150 and len(mm.store._ids) == 150
    hits = mm.store.search(HashingEmbedder().vector("topic149"), k=1)
    assert "149" in hits[0][0].text


def test_recall_balances_relevance_recency_importance(tmp_path):
    mm = make(tmp_path, w_recency=0.35, w_importance=0.5, min_similarity=0.05)
    now = time.time()
    emb = HashingEmbedder()

    def add(text, importance, age_s):
        return mm.store.add("episode", text, emb.vector(text), importance=importance, ts=now - age_s,
                            signature=mm.embedder.signature)

    old = add("bob likes pineapple pizza", 2, 30 * 86400)
    add("bob said pineapple pizza is their favourite food ever", 8, 3600)
    add("the weather is rainy", 9, 3600)
    recalls = run(mm.recall("what pizza does bob like?", k=2, now=now))
    texts = [r.record.text for r in recalls]
    assert "favourite" in texts[0]  # relevant + recent + important beats relevant-but-old
    assert texts[1] == "bob likes pineapple pizza"
    assert all("weather" not in t for t in texts)  # important but irrelevant
    assert mm.store.get_many([old])[old].last_access == now  # recalling refreshes recency
    lines = mm.format_recalls(recalls, now)
    assert lines[0].startswith("(1 h ago)")


def test_recall_skips_just_stored_memories(tmp_path):
    mm = make(tmp_path)
    run(mm.remember("carol loves chess"))
    assert run(mm.recall("chess")) == []  # in working memory already
    assert run(mm.recall("chess", exclude_recent_s=0))


def test_viewer_profiles_and_notes(tmp_path):
    mm = make(tmp_path)
    profile, new = mm.seen_user("Alice", "twitch")
    assert new and profile.messages == 1
    profile, new = mm.seen_user("alice", "twitch")
    assert not new and profile.messages == 2
    run(mm.remember("alice has a cat named Mochi", kind="fact", subject="alice"))
    note = mm.viewer_note("Alice")
    assert "Mochi" in note and "new today" in note
    assert "first message ever" in mm.viewer_note("nobody")


def test_facts_without_llm_store_raw_disclosures(tmp_path):
    mm = make(tmp_path)
    facts = run(mm.extract_facts("dave", "I have a cat named Pixel"))
    assert facts and "Pixel" in facts[0]
    assert run(mm.extract_facts("dave", "lol nice")) == []


def test_working_memory_summarises_overflow(tmp_path):
    mm = make(tmp_path, llm=MockLLM(delay_s=0), history_turns=4, summarize_after=6)

    async def go():
        for i in range(8):
            mm.add_turn(Turn(f"user{i}", f"message {i}"))
        await mm.drain()

    run(go())
    assert len(mm.history) <= 6
    assert mm.summary
    assert mm.store.count("summary") >= 1


def test_observe_reflect_and_end_session_across_streams(tmp_path):
    llm = MockLLM(delay_s=0, seed=3)
    mm = make(tmp_path, llm=llm, reflect_every=5, session="stream-1")

    async def stream_one():
        for i in range(6):
            mm.observe(speaker=f"viewer{i}", said=f"I just finished level {i}", reply="nice!", character="Nexa")
        await mm.drain()
        return await mm.end_session()

    result = run(stream_one())
    assert result["summary"] and result["diary"]
    assert mm.store.count("reflection") >= 1
    mm.store.close()

    # A new stream remembers the old one.
    mm2 = make(tmp_path, llm=llm, session="stream-2")
    assert mm2.last_stream_recap()
    recalls = run(mm2.recall("viewer3 finished level 3", exclude_recent_s=0))
    assert any("level 3" in r.record.text for r in recalls)


def test_importance_and_age_helpers():
    assert estimate_importance("hi") == 3.0
    assert estimate_importance("my name is Sam", support=500, creator=True) > 7
    assert humanize_age(30) == "just now" and humanize_age(3 * 86400) == "3 days ago"
