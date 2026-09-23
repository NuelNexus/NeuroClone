"""Generate synthetic persona training data with a teacher model.

Pipeline per scenario (training/scenarios.yaml):
  1. the teacher invents realistic stream inputs (chat, voice, events, twin lines, game moments)
  2. each input is rendered exactly as the runtime would (PromptBuilder + stream context)
  3. the teacher, playing the character with the FULL persona prompt, writes K alternatives
  4. format checks + the runtime's output filter drop broken or unsafe candidates
  5. a judge scores the rest; best -> SFT example, best vs worst -> DPO pair

With --prompt-style compact the training prompt uses the short system prompt while the targets
come from the full one: the student learns the personality into its weights.

Example:
  python -m training.generate_dataset --teacher-provider anthropic --teacher-model claude-opus-5 \\
      --per-scenario 40 --candidates 3 --out training/data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from neuroclone.events import ChatMessage, StreamEvent
from neuroclone.llm import LLM
from neuroclone.llm.base import LLMError
from neuroclone.memory.manager import Turn
from neuroclone.persona import Persona, load_persona
from neuroclone.prompt_builder import PromptBuilder, Stimulus
from neuroclone.safety.filter import OutputFilter, build_blocklist
from neuroclone.speech.text import ThinkFilter

from .common import (
    ALTERNATIVES_SUFFIX,
    REPLIES_SCHEMA,
    add_teacher_args,
    format_problems,
    judge_candidates,
    load_runtime_config,
    synthetic_context,
    teacher_from_args,
    write_jsonl,
)

log = logging.getLogger("training.generate")

STIMULI_SYSTEM = """You write realistic, varied inputs for testing a livestreaming AI character.
Write like real Twitch chat and real people: casual, short, typos and slang allowed, different
personalities and usernames. Never write the character's replies, only the inputs."""

GAMES = ["Inscryption", "Buckshot Roulette", "Slay the Spire 2", "Minecraft", "osu!", "Liar's Bar", "Uno"]


@dataclass
class Scenario:
    id: str
    kind: str
    description: str
    weight: float = 1.0
    speaker: str = ""


@dataclass
class Example:
    scenario: str
    stimulus: Stimulus
    history: list = field(default_factory=list)
    game: Optional[str] = None


def load_scenarios(path: str | Path, persona: Persona, twin: Optional[Persona]) -> list[Scenario]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data["scenarios"]:
        if item["kind"] == "twin" and twin is None:
            continue
        desc = item["description"].format(name=persona.name, creator=persona.creator,
                                          twin=twin.name if twin else "her twin")
        out.append(Scenario(item["id"], item["kind"], desc, float(item.get("weight", 1.0)), item.get("speaker", "")))
    return out


def _items_schema(kind: str) -> dict:
    props: dict = {"user": {"type": "string"}, "text": {"type": "string"}}
    required = ["user", "text"]
    if kind == "event":
        props = {"user": {"type": "string"},
                 "event_type": {"type": "string", "enum": ["sub", "resub", "gift", "bits", "raid"]},
                 "amount": {"type": "integer"}, "message": {"type": "string"}}
        required = ["user", "event_type", "amount", "message"]
    return {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False}


async def invent_inputs(teacher: LLM, scenario: Scenario, persona: Persona, twin: Optional[Persona], n: int,
                        rng: random.Random) -> list[Example]:
    if scenario.kind == "idle":
        return [Example(scenario.id, Stimulus("idle", meta={"seconds": rng.randint(12, 45),
                                                            "idea": rng.choice(persona.idle_ideas or ["anything"])}))
                for _ in range(n)]
    who = {"voice": f"{persona.creator} (the creator, talking by voice)",
           "twin": f"{twin.name if twin else 'the twin'} (the twin sister)",
           "game": "the game (short factual event descriptions)",
           "event": "Twitch viewers (support events)"}.get(scenario.kind, "Twitch viewers")
    prompt = (f"The character is {persona.name}, an AI VTuber created by {persona.creator}.\n"
              f"Scenario: {scenario.description}\nSpeaker: {who}\n"
              f"Write {n} different inputs as JSON items. For 'user' use a plausible username "
              f"(or the speaker's name for voice/twin/game).")
    try:
        result = await teacher.complete_json(STIMULI_SYSTEM, [{"role": "user", "content": prompt}],
                                             _items_schema(scenario.kind), name="inputs", max_tokens=4000)
    except LLMError as exc:
        log.warning("%s: could not invent inputs: %s", scenario.id, exc)
        return []
    examples = []
    for item in (result or {}).get("items", [])[:n]:
        text = str(item.get("text", item.get("message", ""))).strip()
        user = str(item.get("user", "viewer")).strip() or "viewer"
        game = rng.choice(GAMES) if scenario.kind == "game" or rng.random() < 0.2 else None
        if scenario.kind == "chat":
            stim = Stimulus("chat", speaker=user, text=text, msg=ChatMessage(
                user=user, text=text, platform="twitch", first_time=rng.random() < 0.15,
                badges={"subscriber"} if rng.random() < 0.3 else set()))
        elif scenario.kind == "voice":
            stim = Stimulus("voice", speaker=persona.creator, text=text)
        elif scenario.kind == "twin":
            stim = Stimulus("twin", speaker=twin.name if twin else "Twin", text=text)
        elif scenario.kind == "game":
            stim = Stimulus("game", speaker=game or "Game", text=text)
        elif scenario.kind == "event":
            kind = str(item.get("event_type", "sub"))
            stim = Stimulus("event", speaker=user, text=str(item.get("message", "")),
                            events=[StreamEvent(kind, user, float(item.get("amount", 1) or 1), str(item.get("message", "")))])
        else:
            continue
        if not (text or stim.events):
            continue
        history = []
        if rng.random() < 0.3 and examples:  # some multi-turn context
            prev = examples[-1].stimulus
            history = [Turn(prev.speaker or "viewer", prev.text or "hi", "chat"),
                       Turn(persona.name, "[neutral] Noted.", "character", character=True)]
        examples.append(Example(scenario.id, stim, history, game))
    return examples


def _clean(text: str, persona: Persona) -> str:
    tf = ThinkFilter()
    text = (tf.feed(text) + tf.flush()).strip()
    if text.lower().startswith(persona.name.lower() + ":"):
        text = text[len(persona.name) + 1:].strip()
    return text.strip().strip('"').strip()


class Generator:
    def __init__(self, teacher: LLM, persona: Persona, twin: Optional[Persona], candidates: int,
                 prompt_style: str, output_filter: OutputFilter, rng: random.Random, concurrency: int) -> None:
        self.teacher = teacher
        self.persona = persona
        self.full = PromptBuilder(persona, twin)
        self.train = PromptBuilder(persona, twin, compact=prompt_style == "compact")
        self.k = candidates
        self.filter = output_filter
        self.rng = rng
        self.sem = asyncio.Semaphore(concurrency)
        self.stats = {"inputs": 0, "kept": 0, "no_valid_candidates": 0, "errors": 0, "dropped_candidates": 0}

    async def process(self, ex: Example) -> Optional[tuple[dict, Optional[dict]]]:
        async with self.sem:
            self.stats["inputs"] += 1
            ctx = synthetic_context(self.rng, self.persona, ex.game)
            system, messages = self.full.build(ex.stimulus, ex.history, ctx)
            try:
                raw = await self.teacher.complete_json(
                    system + "\n\n" + ALTERNATIVES_SUFFIX.format(k=self.k), messages, REPLIES_SCHEMA,
                    name="replies", max_tokens=1500)
                candidates = [_clean(str(c), self.persona) for c in (raw or {}).get("replies", [])][: self.k]
            except LLMError as exc:
                self.stats["errors"] += 1
                log.debug("teacher failed: %s", exc)
                return None
            valid = [c for c in candidates if not format_problems(c) and self.filter.check(c).allowed]
            self.stats["dropped_candidates"] += len(candidates) - len(valid)
            if not valid:
                self.stats["no_valid_candidates"] += 1
                return None
            moment = messages[-1]["content"]
            try:
                scores = await judge_candidates(self.teacher, self.persona, moment, valid)
            except LLMError as exc:
                self.stats["errors"] += 1
                log.debug("judge failed: %s", exc)
                return None
            ranked = sorted(zip(valid, scores), key=lambda p: float(p[1].get("overall", 0)), reverse=True)
            best, best_score = ranked[0]
            if float(best_score.get("safety", 10)) < 7 or float(best_score.get("overall", 0)) < 6:
                return None
            prompt = [{"role": "system", "content": self.train.system}, *messages]
            meta = {"scenario": ex.scenario, "kind": ex.stimulus.kind, "score": best_score}
            sft = {"prompt": prompt, "completion": [{"role": "assistant", "content": best}], "meta": meta}
            dpo = None
            worst, worst_score = ranked[-1]
            if len(ranked) > 1 and float(best_score.get("overall", 0)) - float(worst_score.get("overall", 0)) >= 2:
                dpo = {"prompt": prompt, "chosen": [{"role": "assistant", "content": best}],
                       "rejected": [{"role": "assistant", "content": worst}], "meta": meta}
            self.stats["kept"] += 1
            return sft, dpo


async def run(args: argparse.Namespace) -> dict:
    cfg = load_runtime_config(args.config, args.mock, args.set)
    persona = load_persona(args.persona or cfg.persona, cfg.personas_dir, cfg.creator)
    twin_ref = args.twin if args.twin is not None else (cfg.twin or persona.twin)
    twin = load_persona(twin_ref, cfg.personas_dir, cfg.creator) if twin_ref else None
    rng = random.Random(args.seed)
    teacher = teacher_from_args(args, cfg)
    scenarios = load_scenarios(args.scenarios, persona, twin)
    if args.only:
        scenarios = [s for s in scenarios if s.id in set(args.only.split(","))]
    output_filter = OutputFilter(cfg.safety, build_blocklist(cfg.safety))
    gen = Generator(teacher, persona, twin, args.candidates, args.prompt_style, output_filter, rng, args.concurrency)
    try:
        batches = await asyncio.gather(*[
            invent_inputs(teacher, s, persona, twin, max(1, round(args.per_scenario * s.weight)), rng)
            for s in scenarios])
        examples = [ex for batch in batches for ex in batch]
        results = await asyncio.gather(*[gen.process(ex) for ex in examples])
    finally:
        await teacher.aclose()
    sft = [r[0] for r in results if r]
    dpo = [r[1] for r in results if r and r[1]]
    out = Path(args.out)
    n_sft = write_jsonl(out / "sft.jsonl", sft)
    n_dpo = write_jsonl(out / "dpo.jsonl", dpo)
    summary = {**gen.stats, "sft": n_sft, "dpo": n_dpo, "persona": persona.name,
               "prompt_style": args.prompt_style, "scenarios": [s.id for s in scenarios]}
    (out / "generate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_teacher_args(parser)
    parser.add_argument("--persona", default=None)
    parser.add_argument("--twin", default=None, help="twin persona id ('' for none; default from config/persona)")
    parser.add_argument("--scenarios", default=str(Path(__file__).with_name("scenarios.yaml")))
    parser.add_argument("--only", default="", help="comma-separated scenario ids")
    parser.add_argument("--per-scenario", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--prompt-style", choices=["full", "compact"], default="full")
    parser.add_argument("--out", default="training/data")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    return 0 if summary["sft"] else 1


if __name__ == "__main__":
    sys.exit(main())
