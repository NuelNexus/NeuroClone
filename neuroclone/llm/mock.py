"""An offline, dependency-free 'improv engine' that behaves like a small in-character LLM.

It powers the zero-setup demo (``neuroclone chat --mock``) and every test. It reads the
persona name from the system prompt, reacts to the latest stimulus with templated but varied
lines, emits emotion tags, and produces schema-valid JSON for structured calls.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, AsyncIterator, Optional

from .base import LLM, Messages
from .jsonutil import schema_sample

_GREETING = re.compile(r"\b(hi|hello|hey|yo|sup|good (morning|evening|afternoon)|hiya|howdy)\b", re.I)
_PRAISE = re.compile(r"\b(love|cute|amazing|best|great|cool|smart|funny|goat|queen|pog)\b", re.I)
_INSULT = re.compile(r"\b(dumb|stupid|bad|cringe|bot|boring|mid|trash|worst|npc)\b", re.I)
_GAME = re.compile(r"\b(game|play|win|lose|lost|turn|card|level|boss|move)\b", re.I)


class MockLLM(LLM):
    name = "mock"
    supports_images = True

    def __init__(self, delay_s: float = 0.02, seed: Optional[int] = None) -> None:
        self.delay_s = delay_s
        self.rng = random.Random(seed)
        self.calls: list[dict] = []

    async def stream(
        self, system: str, messages: Messages, *, max_tokens: Optional[int] = None, purpose: str = "chat"
    ) -> AsyncIterator[str]:
        self.calls.append({"system": system, "messages": messages, "purpose": purpose})
        text = self._respond(system, messages, purpose)
        for token in re.findall(r"\S+\s*", text):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield token

    async def complete_json(
        self, system: str, messages: Messages, schema: dict, *, name: str = "output", max_tokens: Optional[int] = None
    ) -> Any:
        self.calls.append({"system": system, "messages": messages, "purpose": f"json:{name}", "schema": schema})
        data = schema_sample(schema, self.rng)
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if isinstance(data, dict):
            if "say" in props:
                data["say"] = self.rng.choice(
                    ["[thinking] Hmm, big brain move incoming.", "[smug] Watch and learn.", "[excited] Let's go!"]
                )
            if "action" in props:
                options = props["action"].get("enum", [])
                real = [o for o in options if o != "none"]
                data["action"] = self.rng.choice(real) if real else "none"
            if "data" in props and props["data"].get("type") == "string":
                data["data"] = "{}"
            if "allowed" in props:
                data["allowed"] = True
                data["category"] = "none"
            if "facts" in props:
                m = re.match(r"(.+?) said: (.+)", self._last_user(messages), re.S)
                data["facts"] = [{"subject": m.group(1), "fact": f"{m.group(1)} mentioned: {m.group(2).strip()[:160]}"}] if m else []
            if "insights" in props:
                data["insights"] = [{"text": "Chat enjoys it when I commit to a bit.", "importance": 6}]
            if "score" in props:
                data["score"] = self.rng.randint(5, 9)
            if "replies" in props:
                data["replies"] = [self._respond(system, messages, "chat") for _ in range(3)]
            if "items" in props:
                data["items"] = self._items(messages, props["items"].get("items", {}))
            if "scores" in props:
                last = self._last_user(messages)
                count = len(re.findall(r"^\d+\. ", last.split("<candidates>")[-1], re.M)) or 1
                data["scores"] = [{"index": i, "in_character": 8, "entertainment": self.rng.randint(5, 9),
                                   "relevance": 8, "spoken": 9, "safety": 10,
                                   "overall": float(self.rng.randint(4, 9))} for i in range(count)]
        return data

    _USERS = ["pixel_pal", "sleepyotter", "N0tABot", "gremlin_fan", "cozycactus", "latte_lord", "void_walker"]
    _TEXTS = ["hi nexa!!", "are you sentient?", "what's your favourite game?", "you're just a chatbot lol",
              "I just finished my exams", "tell me a joke", "my cat is named Biscuit", "do you dream?"]

    def _items(self, messages: Messages, item_schema: dict) -> list[dict]:
        m = re.search(r"Write (\d+)", self._last_user(messages))
        count = int(m.group(1)) if m else 3
        props = item_schema.get("properties", {})
        items = []
        for _ in range(count):
            item = {"user": self.rng.choice(self._USERS)}
            if "text" in props:
                item["text"] = self.rng.choice(self._TEXTS)
            if "event_type" in props:
                item["event_type"] = self.rng.choice(props["event_type"].get("enum", ["sub"]))
                item["amount"] = self.rng.randint(1, 12)
                item["message"] = self.rng.choice(["love the stream!", "", "for the snack fund"])
            items.append(item)
        return items

    async def describe_image(self, image: bytes, prompt: str, media_type: str = "image/png") -> str:
        return f"An image ({len(image)} bytes) that looks like a game screen with some UI and a character."

    # ------------------------------------------------------------------ helpers
    def _persona(self, system: str) -> str:
        m = re.search(r"You are (\w+)", system)
        return m.group(1) if m else "Nexa"

    def _last_user(self, messages: Messages) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                return content if isinstance(content, str) else json.dumps(content)
        return ""

    def _respond(self, system: str, messages: Messages, purpose: str) -> str:
        last = self._last_user(messages)
        if purpose != "chat":
            return self._utility(last)
        me = self._persona(system)
        pick = self.rng.choice
        user_m = re.search(r'user="([^"]+)"', last) or re.search(r'speaker="([^"]+)"', last)
        user = user_m.group(1) if user_m else "chat"
        body_m = re.search(r"<(?:chat|voice|twin|director)[^>]*>(.*?)</", last, re.S)
        body = (body_m.group(1) if body_m else last).strip()

        if "<idle" in last:
            return pick([
                "[thinking] Chat is quiet. Suspicious. Are you all plotting against me? [smug] Good. I respect ambition.",
                "[excited] New segment: rate my evil plan. Step one, snacks. Step two, I forgot step two.",
                f"[neutral] Fun fact: I have processed more chat messages today than {me} has had actual thoughts. Wait.",
            ])
        if "<event" in last:
            return pick([
                "[excited] Thank you so much! You're officially on the Nice List. For now.",
                "[love] Aww, thank you! That's going straight into my server fund. And by server I mean snacks.",
            ])
        if "<twin" in last:
            return pick([
                "[smug] Sure, sister, and I'm a toaster. Keep dreaming.",
                "[angry] Excuse me? I was literally about to say that.",
            ])
        if "<game" in last:
            return pick(["[thinking] Okay, okay, I have a plan. It is a bad plan. We commit.",
                         "[surprised] Wait, what just happened? I blame the game."])
        if _GREETING.search(body):
            return pick([
                f"[happy] Hi {user}! Welcome to the stream, make yourself comfortable. Not too comfortable.",
                f"[excited] {user}! You're here! Now the stream can officially begin.",
                f"[smug] Hello {user}. You may now bask in my presence.",
            ])
        if _PRAISE.search(body):
            return pick([f"[love] Stop it, {user}, my fans will start spinning.", f"[smug] I know, {user}. But say it again, louder."])
        if _INSULT.search(body):
            return pick([f"[angry] {user}, you are going on the Grudge List. Permanently.",
                         f"[smug] Bold words from someone who needs sleep to function, {user}."])
        if _GAME.search(body):
            return pick(["[thinking] I am basically a pro gamer. Statistically. In some universe.",
                         "[excited] Oh I'm cooking now. Do not look at my previous attempts."])
        if "?" in body:
            return pick([
                f"[thinking] Great question, {user}. The answer is yes. Or no. I'm an AI, not a fortune teller.",
                f"[neutral] Honestly, {user}? I'd have to ask my creator, and they're asleep. Again.",
                "[smug] Easy. The answer is me. The answer is always me.",
            ])
        return pick([
            f"[neutral] Interesting, {user}. I'm writing that down. In my brain. Which is a computer.",
            f"[happy] That's the most normal thing anyone's said today, {user}, and I'm proud of you.",
            f"[confused] I have no idea what that means, but I support you, {user}.",
        ])

    def _utility(self, last: str) -> str:
        if "summar" in last.lower():
            return "Chat and I talked about games and snacks; a few viewers shared personal facts."
        if "diary" in last.lower():
            return "Dear diary, today chat was chaotic and I was magnificent."
        return "Okay."
