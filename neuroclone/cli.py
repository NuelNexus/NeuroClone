"""Command line interface: ``neuroclone <command>``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import importlib.util
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import Config, ConfigError, apply_mock_profile, load_config

COMMUNITY_BLOCKLIST = (
    "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)


def _load(args) -> Config:
    path = args.config
    if path is None and Path("config/default.yaml").exists():
        path = "config/default.yaml"
    cfg = load_config(path, getattr(args, "set", None))
    if getattr(args, "mock", False):
        cfg = apply_mock_profile(cfg)
    return cfg


def _setup_logging(cfg: Config, verbose: bool) -> None:
    level = logging.DEBUG if verbose else getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def _run(cfg: Config, **kwargs) -> int:
    from .runtime import Runtime

    runtime = Runtime(cfg, **kwargs)
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        pass
    return 0


# ---------------------------------------------------------------------------- commands
def cmd_run(args) -> int:
    cfg = _load(args)
    _setup_logging(cfg, args.verbose)
    return _run(cfg, console=args.console, print_captions=args.console)


def cmd_chat(args) -> int:
    cfg = _load(args)
    if not args.verbose:
        cfg.logging = dataclasses.replace(cfg.logging, level="WARNING")
    _setup_logging(cfg, args.verbose)
    cfg.chat = dataclasses.replace(cfg.chat, twitch=type(cfg.chat.twitch)(), youtube=type(cfg.chat.youtube)())
    if args.no_idle:
        cfg.conductor = dataclasses.replace(cfg.conductor, idle_after_s=0)
    return _run(cfg, console=True, print_captions=True)


def cmd_neuro_api(args) -> int:
    cfg = _load(args)
    _setup_logging(cfg, args.verbose)
    cfg.conductor = dataclasses.replace(cfg.conductor, idle_after_s=0)
    cfg.games = dataclasses.replace(cfg.games, enabled=True, port=args.port or cfg.games.port)
    print(f"Neuro API test server: point your game at ws://{cfg.games.host}:{cfg.games.port} "
          f"(NEURO_SDK_WS_URL). Ctrl+C to stop.", flush=True)
    return _run(cfg, console=args.console, print_captions=True, chat_sources=False)


def cmd_say(args) -> int:
    cfg = _load(args)
    _setup_logging(cfg, args.verbose)

    async def main() -> None:
        from .persona import load_persona
        from .speech.audio import create_player
        from .speech.text import clean_for_tts, extract_tags
        from .speech.tts import create_tts

        persona = load_persona(args.persona or cfg.persona, cfg.personas_dir, cfg.creator)
        tts = create_tts(cfg.tts, persona)
        player = create_player(cfg.audio)
        text, _ = extract_tags(args.text, persona.emotions)
        clip = await tts.synthesize(clean_for_tts(text))
        print(f"{persona.name} ({cfg.tts.provider}): {len(clip.samples)} samples @ {clip.sample_rate} Hz, "
              f"{clip.duration:.2f}s")
        if args.out:
            Path(args.out).write_bytes(clip.to_wav_bytes())
            print(f"wrote {args.out}")
        await player.play(clip)
        await tts.aclose()

    asyncio.run(main())
    return 0


def cmd_persona(args) -> int:
    from .persona import list_builtin_personas, load_persona

    if args.action == "list":
        for name in list_builtin_personas():
            print(name)
        return 0
    cfg = _load(args)
    persona = load_persona(args.id or cfg.persona, cfg.personas_dir, cfg.creator)
    twin = load_persona(cfg.twin, cfg.personas_dir, cfg.creator) if cfg.twin else None
    print(persona.system_prompt(twin))
    return 0


def cmd_memory(args) -> int:
    cfg = _load(args)
    from .memory import MemoryManager, MemoryStore, create_embedder

    store = MemoryStore(cfg.memory.path)
    if args.action == "stats":
        print(json.dumps(store.stats(), indent=2))
        for user in store.top_users(10):
            print(f"  {user.name}: {user.messages} messages over {user.sessions} streams")
        return 0
    if args.action == "forget":
        print(f"forgot {store.delete_subject(args.query)} memories about {args.query}")
        return 0

    async def search() -> None:
        embedder = create_embedder(cfg.memory.embedder, cfg.memory.embed_base_url or cfg.llm.base_url,
                                   cfg.memory.embed_model, cfg.memory.embed_api_key or cfg.llm.api_key)
        mm = MemoryManager(cfg.memory, store, embedder, None, session_id="cli-search")
        for recall in await mm.recall(args.query, k=args.k, exclude_recent_s=0):
            r = recall.record
            print(f"[{r.kind} imp={r.importance:.0f} score={recall.score:.2f}] {r.text}")
        await embedder.aclose()

    asyncio.run(search())
    return 0


def cmd_blocklist(args) -> int:
    import urllib.request

    url = args.url or COMMUNITY_BLOCKLIST
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - user-chosen URL
        text = resp.read().decode("utf-8", "ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    out.write_text("# Downloaded from " + url + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {len(lines)} entries to {out}. Add it to safety.blocklists in your config.")
    return 0


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else '!!'}] {label}{': ' + detail if detail else ''}")
    return ok


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def cmd_doctor(args) -> int:
    print(f"NeuroClone {__version__} doctor")
    try:
        cfg = _load(args)
        source = args.config or ("config/default.yaml" if Path("config/default.yaml").exists() else "built-in defaults")
        _check("config", True, source + (" (+ mock profile)" if args.mock else ""))
    except ConfigError as exc:
        _check("config", False, str(exc))
        return 1
    from .persona import load_persona

    all_ok = True
    try:
        persona = load_persona(cfg.persona, cfg.personas_dir, cfg.creator)
        _check("persona", True, f"{persona.name} (creator {persona.creator})")
        if cfg.twin:
            twin = load_persona(cfg.twin, cfg.personas_dir, cfg.creator)
            _check("twin persona", True, twin.name)
    except Exception as exc:  # noqa: BLE001
        all_ok &= _check("persona", False, str(exc))

    async def probe_llm() -> tuple[bool, str]:
        provider = cfg.llm.provider
        if provider == "mock":
            return True, "mock (offline improv engine)"
        if provider in ("anthropic", "claude"):
            try:
                importlib.import_module("anthropic")
            except ImportError:
                return False, "pip install 'neuroclone[anthropic]'"
            import os

            has_key = bool(cfg.llm.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
            return has_key, "API key found" if has_key else "set ANTHROPIC_API_KEY or llm.api_key"
        import aiohttp

        url = cfg.llm.base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {cfg.llm.api_key}"} if cfg.llm.api_key else {}
        try:
            async with aiohttp.ClientSession(trust_env=True) as s:
                async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status != 200:
                        return False, f"{url} -> HTTP {r.status}"
                    data = await r.json(content_type=None)
            ids = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
            found = cfg.llm.model in ids or any(cfg.llm.model in (i or "") for i in ids)
            return True, f"reachable, model {cfg.llm.model!r} {'found' if found else 'NOT listed (pull it first?)'}"
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach {url} ({exc})"

    ok, detail = asyncio.run(probe_llm())
    all_ok &= _check(f"llm ({cfg.llm.provider})", ok, detail)

    tts_deps = {"edge": ["edge_tts", "miniaudio"], "kokoro": ["kokoro"]}.get(cfg.tts.provider, [])
    missing = [m for m in tts_deps if importlib.util.find_spec(m) is None]
    detail = f"missing {', '.join(missing)}" if missing else cfg.tts.provider
    if cfg.tts.provider == "azure" and not cfg.tts.azure_key:
        missing, detail = ["key"], "set tts.azure_key / AZURE_SPEECH_KEY"
    all_ok &= _check("tts", not missing, detail)

    if importlib.util.find_spec("sounddevice") is None:
        _check("audio output", cfg.audio.player != "sounddevice", "sounddevice not installed (headless playback)")
    else:
        try:
            import sounddevice as sd

            dev = sd.query_devices(kind="output")
            _check("audio output", True, dev.get("name", "default"))
        except Exception as exc:  # noqa: BLE001
            _check("audio output", cfg.audio.player != "sounddevice", f"no output device ({exc})")
    if cfg.stt.enabled:
        all_ok &= _check("speech recognition", importlib.util.find_spec("faster_whisper") is not None,
                         "faster-whisper installed" if importlib.util.find_spec("faster_whisper") else
                         "pip install 'neuroclone[stt]'")
    if cfg.games.enabled:
        all_ok &= _check(f"Neuro API port {cfg.games.port}", _port_free(cfg.games.host, cfg.games.port),
                         "free" if _port_free(cfg.games.host, cfg.games.port) else "in use")
    if cfg.overlay.enabled:
        all_ok &= _check(f"dashboard port {cfg.overlay.port}", _port_free(cfg.overlay.host, cfg.overlay.port),
                         "free" if _port_free(cfg.overlay.host, cfg.overlay.port) else "in use")
    if cfg.avatar.enabled:
        async def probe_vts() -> bool:
            import aiohttp

            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(cfg.avatar.url, timeout=aiohttp.ClientWSTimeout(ws_close=3)):
                        return True
            except Exception:  # noqa: BLE001
                return False

        _check("VTube Studio", asyncio.run(probe_vts()), cfg.avatar.url + " (enable the API in VTS settings)")
    if cfg.memory.enabled:
        from .memory import MemoryStore

        try:
            stats = MemoryStore(cfg.memory.path).stats()
            _check("memory", True, f"{stats['memories']} memories, {stats['users']} viewers, {stats['sessions']} streams")
        except Exception as exc:  # noqa: BLE001
            all_ok &= _check("memory", False, str(exc))
    print("all good" if all_ok else "some checks failed")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroclone", description="Open-source AI VTuber runtime.")
    parser.add_argument("--version", action="version", version=f"neuroclone {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", help="YAML config (default: config/default.yaml if present)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. --set llm.model=qwen3:8b")
        p.add_argument("--mock", action="store_true", help="offline mode: mock LLM, silent TTS, no devices")
        p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("run", help="go live: chat sources, dashboard, Neuro API, voice, avatar")
    common(p)
    p.add_argument("--console", action="store_true", help="also read chat from the terminal and print captions")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("chat", help="talk to the character in your terminal (dashboard still available)")
    common(p)
    p.add_argument("--no-idle", action="store_true", help="don't start conversations when you're quiet")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("neuro-api", help="game-integration test server (like Randy/Jippity, but in character)")
    common(p)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--console", action="store_true")
    p.set_defaults(func=cmd_neuro_api)

    p = sub.add_parser("say", help="speak one line with the configured voice (voice test)")
    common(p)
    p.add_argument("text")
    p.add_argument("--persona", default=None)
    p.add_argument("--out", default=None, help="also save a WAV file")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("doctor", help="check config, models, devices and ports")
    common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("persona", help="list personas or print the rendered character prompt")
    common(p)
    p.add_argument("action", choices=["list", "show"])
    p.add_argument("id", nargs="?", default=None)
    p.set_defaults(func=cmd_persona)

    p = sub.add_parser("memory", help="inspect or edit long-term memory")
    common(p)
    p.add_argument("action", choices=["stats", "search", "forget"])
    p.add_argument("query", nargs="?", default="")
    p.add_argument("-k", type=int, default=8)
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("blocklist", help="download a community word list")
    p.add_argument("action", choices=["fetch"])
    p.add_argument("--url", default=None)
    p.add_argument("--out", default="data/blocklist_community.txt")
    p.set_defaults(func=cmd_blocklist)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
