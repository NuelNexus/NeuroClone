"""``offline: true``: prove that no AI work leaves this PC.

Chat sources (Twitch, YouTube) are allowed: they are the stream itself. Everything that thinks,
speaks, listens, sees or remembers must run locally.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from .config import Config, LLMConfig

CLOUD_LLM_PROVIDERS = {"anthropic", "claude", "openrouter"}
CLOUD_TTS_PROVIDERS = {"azure", "edge"}


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith((".local", ".lan", ".home.arpa")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def runs_locally(cfg: LLMConfig) -> bool:
    """Does this model run on this PC or network (and so share its GPU)?"""
    provider = (cfg.provider or "").lower()
    if provider in CLOUD_LLM_PROVIDERS:
        return False
    return provider == "mock" or is_local_url(cfg.base_url)


def _llm_problems(label: str, cfg: LLMConfig) -> list[str]:
    provider = (cfg.provider or "").lower()
    if provider == "mock":
        return []
    if provider in CLOUD_LLM_PROVIDERS:
        return [f"{label}.provider is {cfg.provider} (a cloud service)"]
    if not is_local_url(cfg.base_url):
        return [f"{label}.base_url {cfg.base_url} is not on this PC or network"]
    return []


def offline_problems(cfg: Config) -> list[str]:
    problems = _llm_problems("llm", cfg.llm)
    if cfg.utility_llm is not None:
        problems += _llm_problems("utility_llm", cfg.utility_llm)
    if cfg.vision.enabled and cfg.vision.llm is not None:
        problems += _llm_problems("vision.llm", cfg.vision.llm)
    tts = (cfg.tts.provider or "").lower()
    if tts in CLOUD_TTS_PROVIDERS:
        problems.append(f"tts.provider is {cfg.tts.provider} (an online voice); use kokoro")
    elif tts == "openai" and not is_local_url(cfg.tts.base_url):
        problems.append(f"tts.base_url {cfg.tts.base_url} is not on this PC or network")
    if cfg.memory.enabled and cfg.memory.embedder in ("openai", "ollama"):
        base = cfg.memory.embed_base_url or cfg.llm.base_url
        if not is_local_url(base):
            problems.append(f"memory embeddings go to {base}, which is not on this PC or network")
    return problems


def enforce_offline() -> None:
    """Stop libraries from phoning home for model files (they must already be downloaded)."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault(var, "1")
