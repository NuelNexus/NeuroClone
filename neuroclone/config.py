"""Configuration: a dataclass tree loaded from YAML.

Strings may reference environment variables as ``${NAME}`` or ``${NAME:-default}``.
Unknown keys raise ``ConfigError`` (typos should never be silently ignored).
"""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | anthropic | mock
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.9
    top_p: float = 0.95
    presence_penalty: float = 0.3
    frequency_penalty: float = 0.2
    max_tokens: int = 350
    timeout_s: float = 60.0
    # JSON strategy for OpenAI-compatible servers: auto tries json_schema, then json_object, then prompt-only.
    json_mode: str = "auto"
    # Anthropic only: effort for live replies vs. structured decisions, and server-side refusal fallbacks.
    effort: str = "low"
    json_effort: str = "medium"
    fallbacks: bool = True
    # OpenAI-compatible only: merged into every request body (e.g. {"chat_template_kwargs": {...}}).
    extra_body: dict = field(default_factory=dict)
    # Mock only: simulated per-token delay.
    mock_delay_s: float = 0.02
    seed: Optional[int] = None


@dataclass
class MemoryConfig:
    enabled: bool = True
    path: str = "data/memory.sqlite3"
    embedder: str = "hash"  # hash | openai
    embed_model: str = "nomic-embed-text"
    embed_base_url: str = ""  # defaults to llm.base_url
    embed_api_key: str = ""
    recall_k: int = 5
    history_turns: int = 24
    summarize_after: int = 40
    reflect_every: int = 30
    extract_facts: bool = True
    recency_half_life_h: float = 72.0
    min_similarity: float = 0.12
    w_relevance: float = 1.0
    w_recency: float = 0.35
    w_importance: float = 0.5


@dataclass
class TwitchConfig:
    channel: str = ""
    nick: str = ""  # empty = anonymous read-only
    oauth_token: str = ""


@dataclass
class YouTubeConfig:
    video_id: str = ""
    api_key: str = ""
    poll_interval_s: float = 6.0  # the Data API quota allows roughly one poll every 5-6 s for a 3 h stream


@dataclass
class ChatConfig:
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    console: bool = True
    ignore_users: list = field(
        default_factory=lambda: ["nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "sery_bot"]
    )


@dataclass
class SelectorConfig:
    max_age_s: float = 45.0
    freshness_half_life_s: float = 20.0
    temperature: float = 0.35  # softmax sampling temperature; 0 = always pick the top score
    user_cooldown_s: float = 60.0
    vibe_min_messages: int = 6
    vibe_window_s: float = 20.0
    seed: Optional[int] = None


@dataclass
class SafetyConfig:
    blocklists: list = field(default_factory=list)
    extra_terms: list = field(default_factory=list)
    use_builtin_lists: bool = True
    max_input_chars: int = 300
    redact_pii: bool = True
    block_links: bool = True
    llm_moderation: bool = False
    moderation_timeout_s: float = 2.5
    moderation_fail_closed: bool = False


@dataclass
class TTSConfig:
    provider: str = "silent"  # silent | azure | openai | edge | kokoro
    voice: str = ""  # empty = persona default for this provider
    rate: str = ""
    pitch: str = ""
    azure_key: str = ""
    azure_region: str = "eastus"
    base_url: str = "http://localhost:8880/v1"
    api_key: str = ""
    model: str = "kokoro"
    kokoro_lang: str = "a"
    chars_per_second: float = 14.0
    timeout_s: float = 20.0


@dataclass
class AudioConfig:
    player: str = "auto"  # auto | sounddevice | null | wav
    device: Optional[str] = None
    wav_dir: str = "data/audio"
    time_scale: float = 1.0


@dataclass
class STTConfig:
    enabled: bool = False
    model: str = "base.en"
    device: str = "auto"
    compute_type: str = "default"
    language: str = "en"
    speaker: str = ""  # defaults to the persona's creator
    input_device: Optional[str] = None
    vad_threshold: float = 0.015
    silence_ms: int = 700
    min_speech_ms: int = 250
    barge_in: bool = True


@dataclass
class AvatarConfig:
    enabled: bool = False
    url: str = "ws://localhost:8001"
    twin_url: str = ""  # a second VTube Studio instance for the twin, e.g. ws://localhost:8002
    plugin_name: str = "NeuroClone"
    plugin_developer: str = "NeuroClone"
    token_path: str = "data/vts_token.txt"
    mouth_param: str = "MouthOpen"
    smile_param: str = "MouthSmile"
    lipsync_gain: float = 1.6
    hotkeys: dict = field(default_factory=dict)  # emotion -> hotkey name or id
    expression_hold_s: float = 6.0


@dataclass
class GamesConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    action_timeout_s: float = 20.0
    max_force_retries: int = 3
    voluntary_actions: bool = True
    voluntary_interval_s: float = 12.0
    context_log_size: int = 40
    random_fallback: bool = True
    voice_chat: bool = True


@dataclass
class OverlayConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    token: str = ""


@dataclass
class ConductorConfig:
    idle_after_s: float = 14.0
    idle_jitter_s: float = 6.0
    max_sentences: int = 4
    reply_gap_s: float = 0.4
    event_batch_s: float = 3.0
    first_clause_chars: int = 48
    banter_chance: float = 0.35
    max_banter: int = 2
    respond_to_game_context: bool = True


@dataclass
class VisionConfig:
    enabled: bool = False
    interval_s: float = 0.0  # 0 = on demand only
    monitor: int = 1
    max_width: int = 1280
    llm: Optional[LLMConfig] = None  # defaults to the main llm


@dataclass
class LoggingConfig:
    level: str = "INFO"
    transcripts_dir: str = "data/transcripts"


@dataclass
class Config:
    persona: str = "nexa"
    twin: str = ""  # optional second persona sharing the stage
    creator: str = ""  # overrides the persona's creator name
    personas_dir: str = "config/personas"
    llm: LLMConfig = field(default_factory=LLMConfig)
    utility_llm: Optional[LLMConfig] = None  # summaries/reflection/moderation; defaults to llm
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    games: GamesConfig = field(default_factory=GamesConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    conductor: ConductorConfig = field(default_factory=ConductorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], where)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return from_dict(tp, value, where)
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: expected an integer, got {value!r}") from exc
    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: expected a number, got {value!r}") from exc
    if tp is str:
        return "" if value is None else str(value)
    if tp is list or origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return list(value)
    if tp is dict or origin is dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)
    return value


def from_dict(cls: type, data: dict, where: str = "config") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid keys are {sorted(names)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def parse_override(text: str) -> dict:
    """Turn ``llm.model=qwen3:8b`` into ``{"llm": {"model": "qwen3:8b"}}`` (value parsed as YAML)."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like section.key=value")
    path, raw = text.split("=", 1)
    value = yaml.safe_load(raw) if raw.strip() else ""
    node: dict = {}
    cur = node
    parts = [p for p in path.strip().split(".") if p]
    for part in parts[:-1]:
        cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return node


def load_config(path: Optional[str | Path] = None, overrides: Optional[list[str]] = None) -> Config:
    data: dict = {}
    if path:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        data = loaded
    for item in overrides or []:
        data = _deep_merge(data, parse_override(item))
    return from_dict(Config, expand_env(data))


def apply_mock_profile(cfg: Config) -> Config:
    """Zero-setup profile: offline mock LLM, silent TTS, no audio device, no external services."""
    cfg.llm = dataclasses.replace(cfg.llm, provider="mock")
    cfg.utility_llm = None
    cfg.tts = dataclasses.replace(cfg.tts, provider="silent")
    cfg.audio = dataclasses.replace(cfg.audio, player="null")
    cfg.memory = dataclasses.replace(cfg.memory, embedder="hash")
    cfg.stt = dataclasses.replace(cfg.stt, enabled=False)
    cfg.avatar = dataclasses.replace(cfg.avatar, enabled=False)
    cfg.vision = dataclasses.replace(cfg.vision, enabled=False)
    cfg.chat = dataclasses.replace(cfg.chat, twitch=TwitchConfig(), youtube=YouTubeConfig())
    return cfg
