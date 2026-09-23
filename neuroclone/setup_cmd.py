"""``neuroclone setup`` and ``neuroclone bench``: tune NeuroClone to this PC, download everything
once, then run fully offline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

from .config import Config, load_config
from .downloads import ProgressBar, ensure_kokoro, ensure_whisper, registry_size_gb
from .hardware import detect
from .planner import CATALOG, EMBED_MODEL, OLLAMA_TUNING, Plan, plan_for, render_config


def _ask(question: str, default: bool, assume: Optional[bool]) -> bool:
    if assume is not None:
        return assume
    if not sys.stdin or not sys.stdin.isatty():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default
    return default if not answer else answer.startswith("y")


def exact_sizes() -> dict[str, float]:
    """Ask the Ollama registry for real download sizes (in parallel; empty when offline)."""
    tags = [m.tag for m in CATALOG]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tags)) as pool:
        sizes = dict(zip(tags, pool.map(registry_size_gb, tags)))
    return {tag: size for tag, size in sizes.items() if size}


# ---------------------------------------------------------------------------- Ollama
async def ollama_status(cfg: Config) -> tuple[bool, str]:
    from .llm.ollama import OllamaLLM

    llm = OllamaLLM(cfg.llm)
    try:
        return True, await llm.version()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        await llm.aclose()


async def pull_models(cfg: Config, tags: list[str]) -> list[str]:
    """``ollama pull`` each tag with a progress bar. Returns the tags that failed."""
    from .llm.ollama import OllamaLLM

    failed = []
    bar = ProgressBar()
    for tag in tags:
        llm = OllamaLLM(dataclasses.replace(cfg.llm, model=tag))
        try:
            if await llm.has_model():
                print(f"  {tag}: already downloaded")
                continue
            await llm.pull(lambda status, done, total, tag=tag: bar(
                f"{tag} {'downloading' if total else status}", done, total))
            bar.done(f"  {tag}: downloaded")
        except Exception as exc:  # noqa: BLE001
            bar.done(f"  {tag}: FAILED ({exc})")
            failed.append(tag)
        finally:
            await llm.aclose()
    return failed


# ---------------------------------------------------------------------------- benchmark
async def benchmark(cfg: Config, *, voice: bool = True) -> dict:
    """Measure what a viewer will feel: model load time, time to first word, speed, voice speed."""
    from .llm import create_llm
    from .persona import load_persona
    from .prompt_builder import PromptBuilder, Stimulus, StreamContext

    out: dict = {}
    persona = load_persona(cfg.persona, cfg.personas_dir, cfg.creator)
    builder = PromptBuilder(persona, compact=cfg.prompt_style == "compact")
    ctx = StreamContext(now=time.time(), uptime_s=1800)
    llm = create_llm(cfg.llm)
    try:
        if hasattr(llm, "warmup"):
            out["load_s"] = await llm.warmup()
        for i, question in enumerate(("hi! how is the stream going?", "what's your favourite snack?")):
            system, messages = builder.build(Stimulus("chat", speaker=f"viewer{i}", text=question), [], ctx, "")
            start = time.monotonic()
            first: Optional[float] = None
            reply = ""
            async for delta in llm.stream(system, messages, max_tokens=120):
                if first is None:
                    first = time.monotonic() - start
                reply += delta
            out[f"first_word_s_{i}"] = first
            out["reply"] = reply.strip()
            stats = getattr(llm, "last_stats", {}) or {}
            if stats.get("tokens_per_s"):
                out["tokens_per_s"] = stats["tokens_per_s"]
            if stats.get("prompt_tokens_per_s"):
                out.setdefault("prompt_tokens_per_s", stats["prompt_tokens_per_s"])
        if hasattr(llm, "residency"):
            out["residency"] = await llm.residency()
    finally:
        await llm.aclose()
    if voice and cfg.tts.provider != "silent":
        from .speech.tts import create_tts

        tts = create_tts(cfg.tts, persona)
        try:
            if hasattr(tts, "warmup"):
                await tts.warmup()
            start = time.monotonic()
            clip = await tts.synthesize("Okay chat, I have a plan. It's a bad plan, but we commit.")
            out["voice_rtf"] = (time.monotonic() - start) / max(clip.duration, 1e-3)
        finally:
            await tts.aclose()
    return out


def print_benchmark(result: dict, plan_split: str = "") -> None:
    first = result.get("first_word_s_1") or result.get("first_word_s_0")
    if result.get("load_s") is not None:
        print(f"  model load:        {result['load_s']:.1f} s (once per session)")
    if first is not None:
        print(f"  first word:        {first:.2f} s after the question (warm)")
    if result.get("tokens_per_s"):
        speed = result["tokens_per_s"]
        verdict = "plenty" if speed >= 15 else "enough for speech" if speed >= 7 else "too slow: pick a smaller model"
        print(f"  generation:        {speed:.0f} tokens/s ({verdict}; speech needs ~5)")
    where = result.get("residency")
    if where:
        print(f"  on the GPU:        {where['gpu_pct']}% of {where['size_gb']:.1f} GB")
        if where["gpu_pct"] < 90:
            print("    part of the model runs on the CPU. For snappier replies re-run setup with --prefer speed")
    if result.get("voice_rtf") is not None:
        rtf = result["voice_rtf"]
        print(f"  voice:             {1 / rtf:.1f}x faster than real time" if rtf < 1 else
              f"  voice:             {rtf:.1f}x SLOWER than real time: lower tts.threads contention or use --prefer speed")
    if result.get("reply"):
        print(f"  sample reply:      {result['reply'][:160]}")


# ---------------------------------------------------------------------------- commands
def cmd_setup(args) -> int:
    print("NeuroClone setup: local, offline and free\n")
    hw = detect()
    print(f"  this PC: {hw.summary()}")
    sizes = {} if args.no_download else exact_sizes()
    plan: Plan = plan_for(hw, prefer=args.prefer, streaming=not args.no_stream, vram_gb=args.vram, model=args.model,
                          sizes=sizes)
    print(f"  plan ({plan.prefer}{', streaming reserve' if plan.streaming else ''}, "
          f"{plan.budget_gb:.1f} GB of VRAM for the brain):")
    for line in plan.describe():
        print(f"    {line}")
    for note in plan.notes:
        print(f"    note: {note}")
    if args.dry_run:
        return 0

    out = Path(args.out)
    mic = _ask("\nTalk to her with your microphone (downloads a speech recogniser)?", False,
               True if args.mic else (False if args.yes else None))
    if out.exists() and not args.force and not _ask(f"{out} exists. Overwrite it?", False,
                                                    False if args.yes else None):
        print(f"  keeping {out} (use --force to regenerate it)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_config(plan, persona=args.persona, creator=args.creator, mic=mic), encoding="utf-8")
        load_config(out)  # fail here, not on stream, if the template ever breaks
        print(f"  wrote {out}")
    cfg = load_config(out)

    if args.no_download or not _ask("Download the models now?", True, args.yes or None):
        print("\nWhen you're ready: neuroclone setup --yes")
        return 0

    ok = True
    print("\nOllama (the brain):")
    up, detail = asyncio.run(ollama_status(cfg))
    if not up:
        ok = False
        print(f"  not running ({detail})")
        print("  install it from https://ollama.com/download (Windows: `winget install Ollama.Ollama`), start it,")
        print("  then run `neuroclone setup --yes` again")
    else:
        print(f"  Ollama {detail} is running")
        failed = asyncio.run(pull_models(cfg, [cfg.llm.model, cfg.memory.embed_model or EMBED_MODEL.tag]))
        ok &= not failed

    print("\nVoice (Kokoro):")
    if importlib.util.find_spec("kokoro_onnx") is None:
        ok = False
        print("  missing package: pip install 'neuroclone[local]'")
    else:
        bar = ProgressBar()
        try:
            fetched = ensure_kokoro(cfg.tts.kokoro_model, cfg.tts.kokoro_voices, progress=bar)
            bar.done("  downloaded " + ", ".join(p.name for p in fetched) if fetched else "  already downloaded")
        except Exception as exc:  # noqa: BLE001
            ok = False
            bar.done(f"  FAILED: {exc}")

    if cfg.stt.enabled:
        print(f"\nEars (Whisper {cfg.stt.model}):")
        if importlib.util.find_spec("faster_whisper") is None:
            ok = False
            print("  missing package: pip install 'neuroclone[local]'")
        else:
            try:
                ensure_whisper(cfg.stt.model, cfg.stt.download_root)
                print("  ready")
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  FAILED: {exc}")

    if up and not args.no_bench:
        print("\nBenchmark (what viewers will feel):")
        try:
            print_benchmark(asyncio.run(benchmark(cfg, voice=importlib.util.find_spec("kokoro_onnx") is not None)))
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  benchmark failed: {exc}")

    print("\nOptional Ollama tuning (set once, then restart Ollama):")
    for name, (value, why) in OLLAMA_TUNING.items():
        print(f"  {name + '=' + value:<30} {why}")
    if sys.platform == "win32":
        print("  (the Windows installer script sets these for you; by hand: setx NAME VALUE)")

    flag = "" if out.as_posix() == "config/local.yaml" else f" -c {out}"
    print("\n" + ("All set. " if ok else "Setup finished with problems (see above). ")
          + f"Start chatting: neuroclone chat{flag}   Go live: neuroclone run --console{flag}")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    from .cli import _load

    cfg = _load(args)
    print(f"Benchmarking {cfg.llm.provider}:{cfg.llm.model} and the {cfg.tts.provider} voice...")
    try:
        print_benchmark(asyncio.run(benchmark(cfg)))
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {exc}")
        return 1
    return 0


def add_parsers(sub, common) -> None:
    p = sub.add_parser("setup", help="detect this PC, pick models, write a config, download everything once")
    p.add_argument("--prefer", choices=["speed", "balanced", "quality"], default="balanced",
                   help="speed = snappiest replies, quality = smartest model that still runs well")
    p.add_argument("--vram", type=float, default=None, help="override the detected GPU memory in GB")
    p.add_argument("--model", default=None, help="use this Ollama model instead of the recommendation")
    p.add_argument("--no-stream", action="store_true", help="not streaming (no OBS/VTube Studio): more VRAM for her")
    p.add_argument("--mic", action="store_true", help="enable microphone input (and download Whisper)")
    p.add_argument("--persona", default="nexa")
    p.add_argument("--creator", default="", help="your name, as the characters should call you")
    p.add_argument("--out", default="config/local.yaml")
    p.add_argument("--force", action="store_true", help="overwrite an existing config")
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to every question")
    p.add_argument("--no-download", action="store_true", help="only write the config")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="only show the plan")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("bench", help="measure first-word latency, tokens/s and voice speed on this PC")
    common(p)
    p.set_defaults(func=cmd_bench)
