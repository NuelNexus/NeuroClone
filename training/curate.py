"""Curate real stream transcripts into training data (the step Neuro-sama's creator reportedly does by hand).

  1. load data/transcripts/*.jsonl written by the runtime
  2. drop filtered, interrupted, errored, badly formatted or unsafe turns
  3. a judge scores each reply against the persona card
  4. good replies (>= --min-score) become SFT examples
  5. weak replies (<= --rewrite-below) get rewritten by the teacher; if the rewrite scores well it
     becomes an SFT example and a DPO pair (chosen = rewrite, rejected = original)
  6. everything lands in review.csv; pass --approved review.csv (with an "approve" column filled in)
     to build datasets only from rows a human signed off

Example:
  python -m training.curate --transcripts "data/transcripts/*.jsonl" --teacher-provider anthropic \\
      --teacher-model claude-opus-5 --out training/data/curated
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from neuroclone.llm.base import LLMError
from neuroclone.persona import load_persona
from neuroclone.safety.filter import OutputFilter, build_blocklist
from neuroclone.speech.text import ThinkFilter, strip_speaker_prefix

from .common import (
    ALTERNATIVES_SUFFIX,
    REPLIES_SCHEMA,
    add_teacher_args,
    format_problems,
    judge_candidates,
    load_runtime_config,
    read_jsonl,
    teacher_from_args,
    write_jsonl,
)

log = logging.getLogger("training.curate")

TRAINABLE_KINDS = {"chat", "voice", "event", "twin", "idle", "vibe", "game", "director"}
APPROVE_VALUES = {"y", "yes", "1", "true", "x", "ok", "approve", "approved"}


def load_transcripts(pattern: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(pattern)):
        for i, rec in enumerate(read_jsonl(path)):
            rec["_id"] = f"{Path(path).stem}:{i}"
            records.append(rec)
    return records


def target_text(rec: dict, name: str) -> str:
    """The raw generation (with emotion tags) is the training target; fall back to what was spoken."""
    raw = rec.get("generated") or rec.get("spoken") or ""
    tf = ThinkFilter()
    return strip_speaker_prefix(tf.feed(raw) + tf.flush(), [name]).strip()


def eligibility(rec: dict, name: str, output_filter: OutputFilter) -> str:
    """Empty string if the record can be used, otherwise the reason it can't."""
    if not isinstance(rec.get("prompt"), list) or not rec["prompt"]:
        return "no_prompt"
    if rec.get("character") and rec["character"] != name:
        return "other_character"
    if (rec.get("stimulus") or {}).get("kind") not in TRAINABLE_KINDS:
        return "kind"
    if rec.get("filtered"):
        return "filtered"
    if rec.get("interrupted"):
        return "interrupted"
    if rec.get("error"):
        return "error"
    text = target_text(rec, name)
    problems = format_problems(text)
    if problems:
        return "format:" + ",".join(problems)
    if not output_filter.check(text).allowed:
        return "unsafe"
    return ""


def load_approvals(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["id"] for row in csv.DictReader(fh) if (row.get("approve") or "").strip().lower() in APPROVE_VALUES}


async def run(args: argparse.Namespace) -> dict:
    cfg = load_runtime_config(args.config, args.mock, args.set)
    persona = load_persona(args.persona or cfg.persona, cfg.personas_dir, cfg.creator)
    twin_ref = cfg.twin or ""
    twin = load_persona(twin_ref, cfg.personas_dir, cfg.creator) if twin_ref else None
    full_system = persona.system_prompt(twin)
    train_system = persona.system_prompt(twin, compact=args.prompt_style == "compact")
    output_filter = OutputFilter(cfg.safety, build_blocklist(cfg.safety))
    approvals = load_approvals(args.approved)
    teacher = teacher_from_args(args, cfg)
    sem = asyncio.Semaphore(args.concurrency)
    records = load_transcripts(args.transcripts)
    stats = {"records": len(records), "eligible": 0, "kept": 0, "rewritten": 0, "dropped": {}}
    rows: list[dict] = []
    sft: list[dict] = []
    dpo: list[dict] = []
    seen_replies: set[str] = set()

    async def handle(rec: dict) -> None:
        reason = eligibility(rec, persona.name, output_filter)
        text = target_text(rec, persona.name)
        if not reason and text.lower() in seen_replies:
            reason = "duplicate"
        if reason:
            stats["dropped"][reason.split(":")[0]] = stats["dropped"].get(reason.split(":")[0], 0) + 1
            return
        seen_replies.add(text.lower())
        stats["eligible"] += 1
        moment = rec["prompt"][-1]["content"] if isinstance(rec["prompt"][-1].get("content"), str) else ""
        async with sem:
            try:
                score = (await judge_candidates(teacher, persona, moment, [text]))[0]
            except LLMError as exc:
                log.warning("judge failed for %s: %s", rec["_id"], exc)
                return
            overall = float(score.get("overall", 0))
            row = {"id": rec["_id"], "character": rec.get("character", persona.name),
                   "stimulus": (rec.get("stimulus") or {}).get("text", "")[:300], "reply": text,
                   "score": overall, "action": "drop", "suggestion": ""}
            prompt = [{"role": "system", "content": train_system}, *rec["prompt"]]
            if overall >= args.min_score:
                row["action"] = "keep"
                if approvals is None or rec["_id"] in approvals:
                    sft.append({"prompt": prompt, "completion": [{"role": "assistant", "content": text}],
                                "meta": {"source": rec["_id"], "score": score}})
                    stats["kept"] += 1
            elif overall <= args.rewrite_below:
                try:
                    raw = await teacher.complete_json(full_system + "\n\n" + ALTERNATIVES_SUFFIX.format(k=2),
                                                      rec["prompt"], REPLIES_SCHEMA, name="replies", max_tokens=1200)
                    options = [str(o).strip() for o in (raw or {}).get("replies", [])][:2]
                    options = [o for o in options if not format_problems(o) and output_filter.check(o).allowed]
                    if options:
                        scores = await judge_candidates(teacher, persona, moment, options)
                        best, best_score = max(zip(options, scores), key=lambda p: float(p[1].get("overall", 0)))
                        if float(best_score.get("overall", 0)) >= args.min_score:
                            row["action"], row["suggestion"] = "rewrite", best
                            if approvals is None or rec["_id"] in approvals:
                                meta = {"source": rec["_id"], "original_score": overall, "score": best_score}
                                sft.append({"prompt": prompt, "completion": [{"role": "assistant", "content": best}],
                                            "meta": meta})
                                dpo.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": best}],
                                            "rejected": [{"role": "assistant", "content": text}], "meta": meta})
                                stats["rewritten"] += 1
                except LLMError as exc:
                    log.warning("rewrite failed for %s: %s", rec["_id"], exc)
            rows.append(row)

    try:
        await asyncio.gather(*[handle(r) for r in records])
    finally:
        await teacher.aclose()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "dpo.jsonl", dpo)
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "character", "stimulus", "reply", "score", "action",
                                                "suggestion", "approve"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["id"]):
            writer.writerow({**row, "approve": ""})
    stats.update({"sft": len(sft), "dpo": len(dpo), "review_csv": str(out / "review.csv")})
    (out / "curate_summary.json").write_text(json.dumps(stats, indent=2))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_teacher_args(parser)
    parser.add_argument("--transcripts", default="data/transcripts/*.jsonl")
    parser.add_argument("--persona", default=None)
    parser.add_argument("--min-score", type=float, default=7.0)
    parser.add_argument("--rewrite-below", type=float, default=5.0)
    parser.add_argument("--prompt-style", choices=["full", "compact"], default="full")
    parser.add_argument("--approved", default=None, help="review.csv with an 'approve' column filled in")
    parser.add_argument("--out", default="training/data/curated")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = asyncio.run(run(build_parser().parse_args(argv)))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
