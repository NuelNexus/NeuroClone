import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from neuroclone.config import LLMConfig
from neuroclone.llm import create_llm
from neuroclone.llm.anthropic_backend import FALLBACK_BETA, AnthropicLLM
from neuroclone.llm.base import LLMError, LLMRefusal
from neuroclone.llm.mock import MockLLM
from neuroclone.llm.ollama import OllamaLLM
from neuroclone.llm.openai_compat import OpenAICompatLLM
from tests.conftest import run


# ---------------------------------------------------------------- OpenAI-compatible
def fake_openai_app(reject_modes=(), stream_tokens=("<think>hmm", "</think>Hel", "lo ", "chat!"), json_text='{"x": 1}'):
    seen: list = []

    async def completions(request):
        body = await request.json()
        seen.append(body)
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for tok in stream_tokens:
                chunk = {"choices": [{"delta": {"content": tok}}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        fmt = (body.get("response_format") or {}).get("type", "prompt")
        if fmt in reject_modes:
            return web.json_response({"error": f"{fmt} unsupported"}, status=400)
        return web.json_response({"choices": [{"message": {"content": json_text}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    return app, seen


async def with_server(app, fn):
    server = TestServer(app)
    await server.start_server()
    try:
        return await fn(f"http://{server.host}:{server.port}/v1")
    finally:
        await server.close()


def test_openai_stream_strips_thinking_and_sends_params():
    app, seen = fake_openai_app()

    async def go(base):
        llm = OpenAICompatLLM(LLMConfig(base_url=base, model="m", temperature=0.5, extra_body={"foo": 1}))
        try:
            return "".join([t async for t in llm.stream("SYS", [{"role": "user", "content": "hi"}])])
        finally:
            await llm.aclose()

    assert run(with_server(app, go)) == "Hello chat!"
    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["temperature"] == 0.5 and body["foo"] == 1 and body["stream"] is True


def test_openai_json_mode_falls_back_and_remembers():
    app, seen = fake_openai_app(reject_modes=("json_schema",), json_text='Sure: {"x": 2}')

    async def go(base):
        llm = OpenAICompatLLM(LLMConfig(base_url=base, model="m"))
        try:
            first = await llm.complete_json("s", [{"role": "user", "content": "q"}], {"type": "object"})
            second = await llm.complete_json("s", [{"role": "user", "content": "q"}], {"type": "object"})
            return first, second, llm.json_mode
        finally:
            await llm.aclose()

    first, second, mode = run(with_server(app, go))
    assert first == second == {"x": 2}
    assert mode == "json_object"
    kinds = [(b.get("response_format") or {}).get("type") for b in seen]
    assert kinds == ["json_schema", "json_object", "json_object"]


def test_openai_unreachable_raises_llm_error():
    llm = OpenAICompatLLM(LLMConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout_s=2))

    async def go():
        try:
            return [t async for t in llm.stream("s", [{"role": "user", "content": "x"}])]
        finally:
            await llm.aclose()

    with pytest.raises(LLMError):
        run(go())


# ---------------------------------------------------------------- Anthropic (fake SDK client)
class FakeStream:
    def __init__(self, texts, final):
        self.texts, self.final = texts, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for t in self.texts:
                yield t
        return gen()

    async def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, texts=("Hi ", "there."), stop_reason="end_turn", json_text='{"ok": true}'):
        self.calls = []
        final = SimpleNamespace(stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber"),
                                content=[SimpleNamespace(type="text", text=json_text)])
        self.texts, self.final = texts, final
        self.beta = SimpleNamespace(messages=self)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.texts, self.final)


def test_anthropic_chat_request_shape():
    client = FakeClient()
    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="claude-opus-5"), client=client)

    async def go():
        return "".join([t async for t in llm.stream("PERSONA", [{"role": "user", "content": "hey"}])])

    assert run(go()) == "Hi there."
    req = client.calls[0]
    assert req["model"] == "claude-opus-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["system"][0]["text"].startswith("PERSONA") and "Latency-sensitive" in req["system"][0]["text"]
    assert "temperature" not in req and "top_p" not in req
    assert req["max_tokens"] >= 2048


def test_anthropic_json_and_no_fallbacks_when_disabled():
    client = FakeClient()
    llm = AnthropicLLM(LLMConfig(provider="anthropic", model="gpt-typo", fallbacks=False), client=client)
    result = run(llm.complete_json("s", [{"role": "user", "content": "q"}], {"type": "object"}))
    assert result == {"ok": True}
    req = client.calls[0]
    assert req["model"] == "claude-opus-5"  # non-Claude model names fall back to the default
    assert req["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert req["output_config"]["effort"] == "medium"
    assert "betas" not in req and "fallbacks" not in req
    assert "Latency-sensitive" not in req["system"][0]["text"]


def test_anthropic_refusal_becomes_llm_refusal():
    llm = AnthropicLLM(LLMConfig(provider="anthropic"), client=FakeClient(stop_reason="refusal"))

    async def go():
        return [t async for t in llm.stream("s", [{"role": "user", "content": "x"}])]

    with pytest.raises(LLMRefusal):
        run(go())


# ---------------------------------------------------------------- mock + factory
def test_mock_stays_in_character_and_makes_valid_json():
    llm = MockLLM(delay_s=0, seed=1)
    text = run(llm.complete("You are Nexa, an AI.", [{"role": "user", "content": '<chat user="amy">hello!</chat>'}],
                            purpose="chat"))
    assert "amy" in text.lower() or "[" in text
    schema = {"type": "object", "properties": {"say": {"type": "string"}, "action": {"type": "string", "enum": ["a", "none"]},
                                               "data": {"type": "string"}}, "required": ["say", "action", "data"]}
    data = run(llm.complete_json("s", [], schema))
    assert data["action"] == "a" and data["data"] == "{}"


def test_factory():
    assert isinstance(create_llm(LLMConfig(provider="mock")), MockLLM)
    assert isinstance(create_llm(LLMConfig(provider="ollama")), OllamaLLM)
    assert isinstance(create_llm(LLMConfig(provider="lmstudio")), OpenAICompatLLM)
    with pytest.raises(LLMError):
        create_llm(LLMConfig(provider="nope"))
