"""GameAgent: turns game context + registered actions into validated action decisions.

Mirrors how Neuro plays through the SDK: the game describes state in text and registers
typed actions; the model picks one high-level action; the game validates and executes it.
We add what the SDK leaves to the backend: schema validation before sending, local retries
with the validation error, retries with the game's failure message, and a last-resort
schema-valid random action so a forced decision can never deadlock the game.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Optional

from ..config import GamesConfig
from ..events import ActionForce
from ..llm.base import LLM, LLMError
from ..llm.jsonutil import JSONExtractError, extract_json, schema_sample, validate
from ..prompts import GAME_SYSTEM_SUFFIX
from .neuro_api import ActionDef, NeuroApiServer

log = logging.getLogger(__name__)


def decision_schema(action_names: list[str], allow_none: bool) -> dict:
    options = list(action_names) + (["none"] if allow_none else [])
    return {
        "type": "object",
        "properties": {
            "say": {"type": "string"},
            "action": {"type": "string", "enum": options},
            "data": {"type": "string"},
        },
        "required": ["say", "action", "data"],
        "additionalProperties": False,
    }


@dataclass
class Decision:
    action: Optional[str]
    data: Optional[str] = None
    say: str = ""
    attempts: int = 1
    fallback: bool = False
    error: str = ""


def _describe_action(action: ActionDef) -> str:
    line = f"- `{action.name}`: {action.description or '(no description)'}"
    if action.schema:
        line += f"\n  parameters (JSON schema): {json.dumps(action.schema, ensure_ascii=False)}"
    else:
        line += "\n  parameters: none (use \"{}\")"
    return line


class GameAgent:
    def __init__(self, llm: LLM, server: NeuroApiServer, persona_system: str, cfg: GamesConfig,
                 rng: Optional[random.Random] = None) -> None:
        self.llm = llm
        self.server = server
        self.system = f"{persona_system}\n\n{GAME_SYSTEM_SUFFIX}"
        self.cfg = cfg
        self.rng = rng or random.Random()

    def build_prompt(self, game: str, actions: list[ActionDef], force: Optional[ActionForce],
                     feedback: list[str], extra_context: str = "") -> str:
        parts = [f"## Game: {game}"]
        history = self.server.context_log(game, limit=15)
        if history:
            parts.append("### What has happened (oldest first)\n" + "\n".join(f"- {h}" for h in history))
        if extra_context:
            parts.append(f"### Also on your mind\n{extra_context}")
        if force is not None:
            if force.state:
                parts.append(f"### Current state\n{force.state}")
            parts.append(f"### Your task right now\n{force.query}")
        else:
            parts.append("### Your task right now\nDecide whether to use one of your actions now. "
                         "Choose \"none\" if nothing useful can be done yet.")
        parts.append("### Actions you can use\n" + "\n".join(_describe_action(a) for a in actions))
        if feedback:
            parts.append("### Your previous attempt failed\n" + "\n".join(f"- {f}" for f in feedback[-3:])
                         + "\nFix the problem and try again.")
        parts.append('Answer with JSON: {"say": short spoken line or "", "action": action name, '
                     '"data": JSON object as a string}.')
        return "\n\n".join(parts)

    def _parse_data(self, action: ActionDef, data) -> tuple[Optional[str], str]:
        """Return (json string or None, error). Accepts a JSON string or an already-parsed object."""
        if action.schema is None:
            return None, ""
        if isinstance(data, dict):
            obj = data
        else:
            text = (data or "").strip() or "{}"
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                try:
                    obj = extract_json(text)
                except JSONExtractError:
                    return None, f"`data` is not valid JSON: {text[:120]}"
        if not isinstance(obj, dict):
            return None, "`data` must be a JSON object"
        error = validate(obj, action.schema)
        if error:
            return None, f"`data` does not match the schema of {action.name}: {error}"
        return json.dumps(obj, ensure_ascii=False), ""

    async def decide(self, game: str, force: Optional[ActionForce] = None, *, allow_none: bool = False,
                     feedback: Optional[list[str]] = None, extra_context: str = "") -> Decision:
        feedback = list(feedback or [])
        registered = {a.name: a for a in self.server.actions(game)}
        names = [n for n in (force.action_names if force else registered) if n in registered]
        if not names:
            return Decision(action=None, error="no actions available")
        actions = [registered[n] for n in names]
        schema = decision_schema(names, allow_none)
        attempts = 0
        for _ in range(2):
            attempts += 1
            prompt = self.build_prompt(game, actions, force, feedback, extra_context)
            try:
                raw = await self.llm.complete_json(self.system, [{"role": "user", "content": prompt}], schema,
                                                   name="game_action", max_tokens=600)
            except LLMError as exc:
                log.warning("game decision failed: %s", exc)
                feedback.append(f"model error: {exc}")
                break
            if not isinstance(raw, dict):
                feedback.append("the answer was not a JSON object")
                continue
            say = str(raw.get("say") or "").strip()
            choice = str(raw.get("action") or "").strip()
            if allow_none and choice in ("", "none"):
                return Decision(action=None, say=say, attempts=attempts)
            if choice not in registered or choice not in names:
                feedback.append(f"'{choice}' is not one of the allowed actions: {', '.join(names)}")
                continue
            data, error = self._parse_data(registered[choice], raw.get("data"))
            if error:
                feedback.append(error)
                continue
            return Decision(action=choice, data=data, say=say, attempts=attempts)
        if force is not None and self.cfg.random_fallback:
            choice = self.rng.choice(names)
            action = registered[choice]
            data = json.dumps(schema_sample(action.schema, self.rng)) if action.schema else None
            log.warning("falling back to a random valid action (%s) after: %s", choice, "; ".join(feedback[-2:]))
            return Decision(action=choice, data=data, say="", attempts=attempts, fallback=True,
                            error="; ".join(feedback[-2:]))
        return Decision(action=None, attempts=attempts, error="; ".join(feedback[-2:]))
