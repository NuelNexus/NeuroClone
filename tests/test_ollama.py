"""The native Ollama backend, the Ollama embedder, and live-priority scheduling."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from neuroclone.config import LLMConfig
from neuroclone.llm.base import LLMError
from neuroclone.llm.mock import MockLLM
from neuroclone.llm.ollama import OllamaLLM, keep_alive_value, native_url
from neuroclone.llm.scheduler import LivePriority, PrioritizedLLM
from neuroclone.memory.embeddings import OllamaEmbedder
from tests.conftest import run


def fake_ollama(tokens=("<think>plan</think>Hi ", "chat", "!"), json_text='{"x": 1}', reject_format=False,
                reject_think=False, models=("qwen3.5:4b",)):
    seen: list = []

    async def chat(request):
        body = await request.json()
        seen.append(body)
        if body.get("model") not in models:
            return web.json_response({"error": f"model '{body.get('model')}' not found"}, status=404)
        if reject_think and "think" in body:
            return web.json_response({"error": "think value not supported for this model"}, status=400)
        if not body.get("messages"):
            return web.json_response({"model": body["model"], "done": True, "done_reason": "load"})
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
            await resp.prepare(request)
            for tok in tokens:
                await resp.write((json.dumps({"message": {"role": "assistant", "content": tok}, "done": False})
                                  + "\n").encode())
            final = {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 30,
                     "eval_duration": 1_000_000_000, "prompt_eval_count": 900, "prompt_eval_duration": 300_000_000}
            await resp.write((json.dumps(final) + "\n").encode())
            return resp
        if reject_format and isinstance(body.get("format"), dict):
            return web.json_response({"error": "invalid JSON schema in format"}, status=400)
        return web.json_response({"message": {"role": "assistant", "content": json_text}, "done": True})

    async def tags(_request):
        return web.json_response({"models": [{"name": m, "model": m} for m in models]})

    async def ps(_request):
        return web.json_response({"models": [{"name": models[0], "model": models[0], "size": 4_000_000_000,
                                              "size_vram": 3_000_000_000, "context_length": 8192}]})

    async def pull(request):
        body = await request.json()
        seen.append(body)
        resp = web.StreamResponse()
        await resp.prepare(request)
        for item in ({"status": "pulling manifest"}, {"status": "pulling abc", "total": 100, "completed": 50},
                     {"status": "pulling abc", "total": 100, "completed": 100}, {"status": "success"}):
            await resp.write((json.dumps(item) + "\n").encode())
        return resp

    async def embed(request):
        body = await request.json()
        seen.append(body)
        return web.json_response({"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    app = web.Application()
    app.router.add_post("/api/chat", chat)
    app.router.add_get("/api/tags", tags)
    app.router.add_get("/api/ps", ps)
    app.router.add_post("/api/pull", pull)
    app.router.add_post("/api/embed", embed)
    return app, seen


async def serve(app, fn):
    server = TestServer(app)
    await server.start_server()
    try:
        return await fn(f"http://{server.host}:{server.port}")
    finally:
        await server.close()


def test_urls_and_keep_alive_parsing():
    assert native_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert native_url("http://box:11434/api/") == "http://box:11434"
    assert native_url("") == "http://localhost:11434"
    assert keep_alive_value("-1") == -1 and keep_alive_value("30m") == "30m" and keep_alive_value(300) == 300


def test_stream_sends_context_think_off_and_keep_alive():
    app, seen = fake_ollama()

    async def go(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base + "/v1", model="qwen3.5:4b", num_ctx=8192,
                                  num_gpu=99, keep_alive="-1", max_tokens=200))
        try:
            text = "".join([t async for t in llm.stream("SYS", [{"role": "user", "content": "hi"}])])
            return text, llm.last_stats
        finally:
            await llm.aclose()

    text, stats = run(serve(app, go))
    assert text == "Hi chat!"  # <think> content never reaches the voice
    body = seen[0]
    assert body["think"] is False and body["keep_alive"] == -1 and body["stream"] is True
    assert body["options"]["num_ctx"] == 8192 and body["options"]["num_gpu"] == 99
    assert body["options"]["num_predict"] == 200
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert stats["tokens_per_s"] == pytest.approx(30) and stats["prompt_tokens"] == 900


def test_think_flag_dropped_when_rejected():
    app, seen = fake_ollama(reject_think=True)

    async def go(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base, model="qwen3.5:4b"))
        try:
            return "".join([t async for t in llm.stream("S", [{"role": "user", "content": "hi"}])]), llm.think
        finally:
            await llm.aclose()

    text, think = run(serve(app, go))
    assert text == "Hi chat!" and think is None and "think" not in seen[-1]


def test_json_uses_schema_then_falls_back():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    app, seen = fake_ollama(json_text='{"x": 7}')

    async def go(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base, model="qwen3.5:4b"))
        try:
            return await llm.complete_json("S", [{"role": "user", "content": "q"}], schema)
        finally:
            await llm.aclose()

    assert run(serve(app, go)) == {"x": 7}
    assert seen[0]["format"] == schema and seen[0]["stream"] is False

    app, seen = fake_ollama(json_text='sure: {"x": 8}', reject_format=True)

    async def go2(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base, model="qwen3.5:4b"))
        try:
            return await llm.complete_json("S", [{"role": "user", "content": "q"}], schema), llm.json_mode
        finally:
            await llm.aclose()

    result, mode = run(serve(app, go2))
    assert result == {"x": 8} and mode == "json" and seen[-1]["format"] == "json"


def test_missing_model_and_unreachable_server_explain_the_fix():
    app, _ = fake_ollama()

    async def go(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base, model="gemma4:12b"))
        try:
            with pytest.raises(LLMError, match="ollama pull gemma4:12b"):
                [t async for t in llm.stream("S", [{"role": "user", "content": "hi"}])]
        finally:
            await llm.aclose()

    run(serve(app, go))

    async def down():
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url="http://127.0.0.1:9", model="m", timeout_s=2))
        try:
            with pytest.raises(LLMError, match="Start the Ollama app"):
                [t async for t in llm.stream("S", [{"role": "user", "content": "hi"}])]
        finally:
            await llm.aclose()

    run(down())


def test_model_management_images_and_embeddings():
    app, seen = fake_ollama()
    progress = []

    async def go(base):
        llm = OllamaLLM(LLMConfig(provider="ollama", base_url=base, model="qwen3.5:4b", num_ctx=6144, num_gpu=20))
        emb = OllamaEmbedder(base, "nomic-embed-text", on_cpu=True)
        try:
            load_s = await llm.warmup()
            has = await llm.has_model()
            where = await llm.residency()
            await llm.pull(lambda status, done, total: progress.append((status, done, total)))
            await llm.describe_image(b"\x89PNG", "what is this?")
            vecs = await emb.embed(["a", "b"])
            return load_s, has, where, vecs
        finally:
            await llm.aclose()
            await emb.aclose()

    load_s, has, where, vecs = run(serve(app, go))
    warm = seen[0]
    assert warm["messages"] == [] and warm["options"] == {"num_ctx": 6144, "num_gpu": 20}  # same load options
    assert has and where["gpu_pct"] == 75 and load_s >= 0
    assert ("success", 0, 0) in progress and any(done == 100 for _, done, _ in progress)
    image_call = next(b for b in seen if isinstance(b, dict) and b.get("messages") and "images" in b["messages"][-1])
    assert image_call["messages"][-1]["images"] == ["iVBORw=="]
    embed_call = next(b for b in seen if "input" in b)
    assert embed_call["options"] == {"num_gpu": 0}  # embeddings stay off the GPU
    assert vecs.shape == (2, 2) and abs(float(vecs[0] @ vecs[0]) - 1) < 1e-6


# ---------------------------------------------------------------- live priority
class SlowLLM(MockLLM):
    def __init__(self):
        super().__init__(delay_s=0)
        self.started = 0
        self.finished = 0

    async def complete_json(self, system, messages, schema, *, name="output", max_tokens=None):
        self.started += 1
        await asyncio.sleep(0.2)
        self.finished += 1
        return {"ok": True}


def test_background_waits_for_live_and_is_preempted_then_retried():
    async def go():
        gate = LivePriority(background_delay_s=0.05)
        inner = SlowLLM()
        live = PrioritizedLLM(inner, gate, live=True)
        background = PrioritizedLLM(inner, gate, live=False)

        bg = asyncio.ensure_future(background.complete_json("s", [], {"type": "object"}))
        await asyncio.sleep(0.05)  # background call is running...
        assert inner.started == 1
        tokens = [t async for t in live.stream("SYS", [{"role": "user", "content": "hi"}])]  # ...a reply preempts it
        assert tokens
        result = await asyncio.wait_for(bg, 3)
        return result, inner.started, inner.finished, gate

    result, started, finished, gate = run(go())
    assert result == {"ok": True} and gate.preempted == 1 and not gate.live_active
    assert started == 2 and finished == 1  # the interrupted call was retried after the reply


def test_background_does_not_start_during_a_live_reply():
    async def go():
        gate = LivePriority(background_delay_s=0.0)
        inner = SlowLLM()
        background = PrioritizedLLM(inner, gate, live=False)
        async with gate.live():
            bg = asyncio.ensure_future(background.complete_json("s", [], {}))
            await asyncio.sleep(0.1)
            assert inner.started == 0  # waiting, not competing for the GPU
        return await asyncio.wait_for(bg, 3), inner.started

    assert run(go()) == ({"ok": True}, 1)


def test_background_is_not_starved_forever():
    async def go():
        gate = LivePriority(background_delay_s=0.0, max_preemptions=2)
        inner = SlowLLM()
        background = PrioritizedLLM(inner, gate, live=False)
        bg = asyncio.ensure_future(background.complete_json("s", [], {}))
        for _ in range(4):
            await asyncio.sleep(0.03)
            async with gate.live():
                await asyncio.sleep(0.01)
        return await asyncio.wait_for(bg, 3), gate.preempted

    result, preempted = run(go())
    assert result == {"ok": True} and preempted == 2


def test_cancelling_the_caller_cancels_the_background_call():
    async def go():
        gate = LivePriority(background_delay_s=0.0)
        inner = SlowLLM()
        bg = asyncio.ensure_future(PrioritizedLLM(inner, gate, live=False).complete_json("s", [], {}))
        await asyncio.sleep(0.05)
        bg.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bg
        await asyncio.sleep(0.3)
        return inner.finished

    assert run(go()) == 0


# ---------------------------------------------------------------- the whole runtime on Ollama
def test_runtime_runs_offline_on_ollama(tmp_cfg):
    from neuroclone.events import ChatMessage
    from tests.test_conductor import Live

    app, seen = fake_ollama(models=("qwen3.5:4b", "nomic-embed-text"))

    async def go(base):
        tmp_cfg.offline = True
        tmp_cfg.llm = LLMConfig(provider="ollama", base_url=base, model="qwen3.5:4b", num_ctx=8192)
        tmp_cfg.memory.embedder = "ollama"
        tmp_cfg.memory.embed_model = "nomic-embed-text"
        tmp_cfg.memory.embed_base_url = base
        async with Live(tmp_cfg) as live:
            live.rt.submit(ChatMessage(user="alice", text="hi nexa, how is the stream?", platform="twitch"))
            turns = await live.wait_turns(1)
            await live.rt.memory.drain()
            return turns

    turns = run(serve(app, go))
    assert turns[0]["reply"] == "Hi chat!"
    chats = [b for b in seen if b.get("model") == "qwen3.5:4b"]
    assert chats[0]["messages"] == []  # warm-up loaded the model before the first viewer
    live_call = next(b for b in chats if b.get("stream"))
    assert live_call["think"] is False and live_call["options"]["num_ctx"] == 8192
    assert all(b["options"]["num_ctx"] == 8192 for b in chats)  # same load options: no reloads
    embeds = [b for b in seen if "input" in b]
    assert embeds and all(b["options"] == {"num_gpu": 0} for b in embeds)
