"""Turn "this PC" into "these models, these settings": the brain of ``neuroclone setup``.

Everything runs locally and for free: the LLM in Ollama on the GPU, the voice (Kokoro) and the
ears (Whisper) on the CPU, memory embeddings on the CPU. The GPU is the scarce resource, so the
plan gives it to the LLM and keeps a reserve for Windows, OBS and VTube Studio while streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hardware import Hardware


@dataclass(frozen=True)
class ModelChoice:
    tag: str  # Ollama model name
    size_gb: float  # approximate download / VRAM size of the default quantisation
    vision: bool
    note: str
    moe: bool = False  # mixture-of-experts: few weights are active per token, so CPU offload hurts less


# Best first. Sizes are estimates; `neuroclone setup` asks the Ollama registry for exact sizes
# when it can, and checks the real GPU/CPU split after loading.
CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice("gemma4:31b", 18.5, True, "Gemma 4 31B: the smartest option, for 24 GB cards"),
    ModelChoice("gemma4:26b", 15.5, True, "Gemma 4 26B mixture-of-experts (4B active): fast for its size", moe=True),
    ModelChoice("gemma4:12b", 7.4, True, "Gemma 4 12B: great conversational quality"),
    ModelChoice("qwen3.5:9b", 6.6, True, "Qwen 3.5 9B: sharp and quick"),
    ModelChoice("gemma4:e4b", 5.0, True, "Gemma 4 E4B: built for small GPUs"),
    ModelChoice("qwen3.5:4b", 3.4, True, "Qwen 3.5 4B: light, fast, still fun"),
    ModelChoice("qwen3.5:2b", 2.2, True, "Qwen 3.5 2B: for 4 GB cards or CPU-only PCs"),
    ModelChoice("qwen3.5:0.8b", 1.0, True, "Qwen 3.5 0.8B: last resort"),
)
EMBED_MODEL = ModelChoice("nomic-embed-text", 0.3, False, "memory search (runs on the CPU)")
PREFERENCES = ("speed", "balanced", "quality")


def find_model(tag: str) -> ModelChoice:
    for m in CATALOG:
        if m.tag == tag:
            return m
    return ModelChoice(tag, 0.0, False, "custom model")


@dataclass
class Plan:
    hardware: Hardware
    prefer: str
    streaming: bool
    model: ModelChoice
    on_gpu: bool
    budget_gb: float  # VRAM the LLM may use
    num_ctx: int
    history_turns: int
    num_thread: int
    tts_threads: int
    stt_model: str
    stt_threads: int
    notes: list[str] = field(default_factory=list)

    @property
    def expected_split(self) -> str:
        if not self.on_gpu:
            return "on the CPU"
        need = self.model.size_gb + kv_overhead(self.num_ctx)
        if need <= self.budget_gb:
            return "fully on the GPU"
        share = max(5, min(95, int(100 * self.budget_gb / need)))
        return f"~{share}% on the GPU, rest on the CPU"

    def downloads(self) -> list[tuple[str, float]]:
        return [(self.model.tag, self.model.size_gb), (EMBED_MODEL.tag, EMBED_MODEL.size_gb),
                ("Kokoro voice", 0.35), (f"Whisper {self.stt_model}", _WHISPER_GB.get(self.stt_model, 0.5))]

    def describe(self) -> list[str]:
        cpu_note = f"{self.tts_threads} CPU threads"
        lines = [
            f"brain:   {self.model.tag} ({self.model.note}), {self.expected_split}, "
            f"{self.num_ctx // 1024}k context, thinking off",
            f"memory:  {EMBED_MODEL.tag} on the CPU",
            f"voice:   Kokoro, offline on the CPU ({cpu_note}), so the GPU stays free for the brain",
            f"ears:    Whisper {self.stt_model} on the CPU (int8)",
            "eyes:    " + ("the same model can see your screen (on demand)" if self.model.vision else "off"),
        ]
        total = sum(size for _, size in self.downloads())
        lines.append(f"download: ~{total:.1f} GB once; after that everything runs offline")
        return lines


_WHISPER_GB = {"base.en": 0.15, "small.en": 0.5, "distil-small.en": 0.35, "medium.en": 1.5}


def kv_overhead(num_ctx: int) -> float:
    """Context cache + runtime overhead in GB (q8 KV cache; hybrid-attention models need even less)."""
    return 0.5 + 0.35 * num_ctx / 8192


def reserve_gb(streaming: bool) -> float:
    """VRAM left for the desktop, OBS (NVENC) and VTube Studio while streaming."""
    return 1.5 if streaming else 0.8


def plan_for(hw: Hardware, *, prefer: str = "balanced", streaming: bool = True, vram_gb: Optional[float] = None,
             model: Optional[str] = None, sizes: Optional[dict] = None) -> Plan:
    """Pick models and settings. ``sizes`` overrides catalog sizes (e.g. exact ones from the registry)."""
    if prefer not in PREFERENCES:
        raise ValueError(f"prefer must be one of {PREFERENCES}")
    catalog = [ModelChoice(m.tag, (sizes or {}).get(m.tag) or m.size_gb, m.vision, m.note, m.moe) for m in CATALOG]
    gpu = hw.gpu
    vram = vram_gb if vram_gb is not None else (gpu.vram_gb if gpu is not None else 0.0)
    if gpu is not None and gpu.vendor == "apple" and vram_gb is None:
        vram = max(0.0, vram - 2.0)  # unified memory: leave room for macOS
    on_gpu = vram >= 3.5
    budget = max(0.0, vram - reserve_gb(streaming)) if on_gpu else 0.0
    notes: list[str] = []

    physical = max(1, hw.cores)
    tts_threads = max(2, min(4, physical // 2))
    stt_model = "small.en" if physical >= 6 else "base.en"
    if prefer == "speed" and stt_model == "small.en":
        stt_model = "base.en"

    if on_gpu:
        num_ctx, history = 8192, 24
        floor = "qwen3.5:0.8b" if prefer == "speed" else "qwen3.5:2b"
        choice = None
        for m in catalog:
            # How far past the VRAM budget a model may go (the rest runs on the CPU from system RAM).
            # Dense models slow down sharply when offloaded; mixture-of-experts models much less.
            if prefer == "quality" and hw.ram_gb >= 24:
                factor = 1.35 if m.moe else 1.15
            else:
                factor = {"speed": 0.85, "balanced": 1.0, "quality": 1.1}[prefer]
            if m.size_gb + kv_overhead(num_ctx) <= budget * factor:
                choice = m
                break
            if m.tag == floor:
                choice = m
                break
        choice = choice or next(m for m in catalog if m.tag == floor)
        num_thread = 0  # let Ollama decide for the few layers that may land on the CPU
        if prefer == "quality" and choice.size_gb + kv_overhead(num_ctx) > budget:
            notes.append("quality mode lets part of the model run on the CPU: smarter, but slower first words")
        if gpu is not None and gpu.vendor in ("amd", "intel"):
            notes.append(f"{gpu.vendor.upper()} GPU: Ollama uses ROCm or Vulkan; if it falls back to the CPU, "
                         "update the graphics driver")
    else:
        num_ctx, history = 4096, 14
        cpu_tag = "qwen3.5:4b" if prefer == "quality" and physical >= 8 else "qwen3.5:2b"
        choice = next(m for m in catalog if m.tag == cpu_tag)
        num_thread = physical
        notes.append("no usable GPU: the brain runs on the CPU; expect a second or two before each reply")

    if model:
        choice = find_model(model)
        notes.append(f"model chosen by hand: {model}")
    if hw.ram_gb and hw.ram_gb < 12:
        notes.append("less than 12 GB RAM: close other apps while streaming")
    return Plan(hw, prefer, streaming, choice, on_gpu, round(budget, 1), num_ctx, history, num_thread,
                tts_threads, stt_model, min(4, physical), notes)


def render_config(plan: Plan, *, persona: str = "nexa", creator: str = "", mic: bool = False) -> str:
    """A commented YAML config for this plan (loads with ``neuroclone.config.load_config``)."""
    hw = plan.hardware
    m = plan.model
    num_gpu = "" if plan.on_gpu else "\n  num_gpu: 0              # no usable GPU: run on the CPU"
    num_thread = f"\n  num_thread: {plan.num_thread}" if plan.num_thread else ""
    vision = "true" if m.vision else "false"
    return f"""# NeuroClone local config, written by `neuroclone setup` for:
#   {hw.summary()}
# Everything below runs on this PC, offline and free. Re-run `neuroclone setup` after a hardware change,
# or edit by hand (every key is documented in neuroclone/config.py).

persona: {persona}
creator: "{creator}"
offline: true              # refuse cloud AI services; chat sources (Twitch/YouTube) still work

llm:
  provider: ollama         # native Ollama API: sets the context size, keeps the model loaded, thinking off
  base_url: http://localhost:11434
  model: {m.tag}
  num_ctx: {plan.num_ctx}           # the persona + memories + chat history fit comfortably
  keep_alive: 30m
  think: false             # thinking models answer seconds faster without it
  temperature: 0.9
  max_tokens: 300{num_gpu}{num_thread}

memory:
  embedder: ollama
  embed_model: {EMBED_MODEL.tag}
  embed_on_cpu: true       # tiny model; keeping it off the GPU avoids evicting the brain
  history_turns: {plan.history_turns}

safety:
  llm_moderation: false    # the rule-based filters stay on; a second LLM pass would compete for the GPU

tts:
  provider: kokoro         # offline neural voice (Apache-2.0)
  kokoro_model: models/kokoro/kokoro-v1.0.onnx
  kokoro_voices: models/kokoro/voices-v1.0.bin
  device: cpu              # faster than real time on the CPU; the GPU is busy thinking
  threads: {plan.tts_threads}
  pitch_semitones: 0       # +1..+3 for a brighter voice (the persona adds its own shift)

audio:
  player: auto

stt:
  enabled: {"true " if mic else "false"}           # talk to her with your microphone
  model: {plan.stt_model}
  device: cpu
  compute_type: int8
  cpu_threads: {plan.stt_threads}
  download_root: models/whisper

vision:
  enabled: {vision}            # the brain model can see: the dashboard's "Look at screen" button (or /look)
  interval_s: 0            # 0 = only when asked; e.g. 30 = glance every 30 s (costs GPU time)

performance:
  live_priority: true      # background memory work never delays a live reply
  warmup: true             # load every model at startup, not on the first message

avatar:
  enabled: false           # VTube Studio: Settings > Start API (port 8001), then set true
  hotkeys: {{}}

overlay:
  enabled: true            # dashboard http://127.0.0.1:8080, OBS captions http://127.0.0.1:8080/overlay
"""


OLLAMA_TUNING = {
    "OLLAMA_KV_CACHE_TYPE": ("q8_0", "halves the context cache memory with no audible quality loss"),
    "OLLAMA_FLASH_ATTENTION": ("1", "needed for the q8_0 cache; faster on most GPUs"),
    "OLLAMA_MAX_LOADED_MODELS": ("2", "the brain and the memory embedder stay loaded together"),
}
