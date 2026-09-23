"""Running on your own PC: hardware detection, the model planner, offline mode, the local voice."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from neuroclone import hardware
from neuroclone.cli import main
from neuroclone.config import Config, ConfigError, LLMConfig, load_config
from neuroclone.emotion import VoiceStyle
from neuroclone.hardware import GPU, Hardware
from neuroclone.offline import is_local_url, offline_problems
from neuroclone.planner import CATALOG, plan_for, render_config
from neuroclone.speech.tts import (
    KokoroTTS,
    TTSError,
    apply_ratio,
    parse_voice_mix,
    pitch_ratio,
    resample_to,
    speed_and_pitch,
)
from tests.conftest import run


def pc(vram=None, ram=32, cores=6, vendor="nvidia"):
    gpus = [GPU("NVIDIA GeForce RTX" if vendor == "nvidia" else "AMD Radeon RX", vendor, vram, vram)] if vram else []
    return Hardware("windows", "11", "Intel Core i5-12400F", cores, cores * 2, ram, gpus)


# ---------------------------------------------------------------- hardware
def test_nvidia_smi_parsing(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware, "_run", lambda cmd, timeout=8.0: "NVIDIA GeForce RTX 4060, 8188, 7400, 576.02\n")
    gpus = hardware.nvidia_gpus()
    assert gpus == [GPU("NVIDIA GeForce RTX 4060", "nvidia", 8188 / 1024, 7400 / 1024, "576.02")]
    assert pc(8).gpu is not None and not GPU("Intel UHD", "intel", 0.1).usable


def test_detect_never_raises_and_summarises():
    hw = hardware.detect()
    assert hw.cores >= 1 and hw.threads >= 1 and hw.os in ("windows", "linux", "macos")
    assert "RAM" in hw.summary()


# ---------------------------------------------------------------- planner
@pytest.mark.parametrize("vram,expected", [
    (None, "qwen3.5:2b"), (4, "qwen3.5:2b"), (6, "qwen3.5:4b"), (8, "gemma4:e4b"), (12, "gemma4:12b"),
    (24, "gemma4:31b"),
])
def test_balanced_plan_fits_the_gpu(vram, expected):
    plan = plan_for(pc(vram))
    assert plan.model.tag == expected
    if vram and vram >= 6:
        assert plan.expected_split == "fully on the GPU"


def test_plan_preferences_streaming_reserve_and_overrides():
    assert plan_for(pc(16), prefer="quality").model.moe  # offloading a mixture-of-experts model is cheap
    assert plan_for(pc(8), prefer="speed").model.tag == "qwen3.5:4b"
    # Not streaming leaves more VRAM for the model.
    assert plan_for(pc(8), streaming=False).budget_gb > plan_for(pc(8)).budget_gb
    assert plan_for(pc(8), model="llama3.2:3b").model.tag == "llama3.2:3b"
    # Exact registry sizes win over the catalog estimates.
    bigger = {"gemma4:e4b": 9.9}
    assert plan_for(pc(8), sizes=bigger).model.tag == "qwen3.5:4b"
    cpu = plan_for(pc(None))
    assert not cpu.on_gpu and cpu.num_ctx == 4096 and cpu.num_thread == 6
    with pytest.raises(ValueError):
        plan_for(pc(8), prefer="turbo")
    assert [m.tag for m in CATALOG][0] == "gemma4:31b"


def test_rendered_config_loads_and_is_offline(tmp_path):
    plan = plan_for(pc(12))
    path = tmp_path / "local.yaml"
    path.write_text(render_config(plan, creator="Nuel", mic=True), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.offline and cfg.llm.provider == "ollama" and cfg.llm.model == "gemma4:12b"
    assert cfg.llm.num_ctx == 8192 and cfg.llm.think is False and cfg.stt.enabled
    assert cfg.tts.provider == "kokoro" and cfg.memory.embedder == "ollama" and cfg.vision.enabled
    assert offline_problems(cfg) == []
    cpu_cfg_path = tmp_path / "cpu.yaml"
    cpu_cfg_path.write_text(render_config(plan_for(pc(None))), encoding="utf-8")
    assert load_config(cpu_cfg_path).llm.num_gpu == 0


def test_setup_dry_run_cli(capsys):
    assert main(["setup", "--dry-run", "--no-download", "--vram", "8"]) == 0
    out = capsys.readouterr().out
    assert "gemma4:e4b" in out and "Kokoro" in out and "offline" in out


# ---------------------------------------------------------------- offline mode
def test_offline_guard():
    assert is_local_url("http://localhost:11434") and is_local_url("http://192.168.1.20:11434")
    assert not is_local_url("https://api.openai.com/v1")
    cfg = Config(offline=True)
    assert offline_problems(cfg) == []
    cfg.llm = LLMConfig(provider="anthropic")
    cfg.tts.provider = "azure"
    problems = offline_problems(cfg)
    assert any("anthropic" in p for p in problems) and any("azure" in p for p in problems)


def test_runtime_refuses_cloud_services_when_offline(tmp_cfg):
    from neuroclone.runtime import Runtime

    tmp_cfg.offline = True
    tmp_cfg.tts.provider = "edge"
    with pytest.raises(ConfigError, match="online voice"):
        run(Runtime(tmp_cfg).setup())


# ---------------------------------------------------------------- local voice
def test_voice_mix_and_pitch_helpers():
    assert parse_voice_mix("af_bella") == [("af_bella", 1.0)]
    assert parse_voice_mix("af_bella:3+af_sky:1") == [("af_bella", 0.75), ("af_sky", 0.25)]
    with pytest.raises(TTSError):
        parse_voice_mix("+")
    assert pitch_ratio(12) == pytest.approx(2.0)
    assert pitch_ratio(0, VoiceStyle(pitch_pct=10)) == pytest.approx(1.1)
    assert speed_and_pitch(1.0, 1.25) == pytest.approx(0.8) and speed_and_pitch(0.6, 2.0) == 0.5


def test_resampling_raises_pitch_and_keeps_length():
    sr = 24000
    t = np.arange(int(1.2 * sr)) / sr
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)  # what the model says 1.2x slower...
    out = apply_ratio(tone, 1.2)  # ...played back at normal speed
    assert len(out) == round(len(tone) / 1.2)
    peak = np.argmax(np.abs(np.fft.rfft(out))) * sr / len(out)
    assert peak == pytest.approx(240, abs=3)
    assert len(resample_to(tone, 100)) == 100 and resample_to(tone[:0], 10).size == 0


class FakeKokoro:
    created: list = []

    @classmethod
    def from_session(cls, session, voices_path):
        return cls()

    def get_voices(self):
        return ["af_bella", "af_sky", "bf_emma"]

    def get_voice_style(self, name):
        return np.full((510, 1, 256), {"af_bella": 1.0, "af_sky": 3.0}.get(name, 0.0), dtype=np.float32)

    def create(self, text, voice, speed=1.0, lang="en-us"):
        FakeKokoro.created.append((text, voice, speed, lang))
        return np.zeros(int(24000 * 0.5 / speed), dtype=np.float32), 24000


@pytest.fixture
def fake_kokoro(monkeypatch, tmp_path):
    ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=lambda *a, **k: None,
                                get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    model, voices = tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    FakeKokoro.created = []
    return str(model), str(voices)


def test_kokoro_tts_pitch_blend_language_and_missing_files(fake_kokoro, tmp_path):
    model, voices = fake_kokoro
    tts = KokoroTTS("af_bella:0.5+af_sky:0.5", model_path=model, voices_path=voices, speed=1.0, pitch_semitones=12)
    clip = run(tts.synthesize("hello", VoiceStyle(rate_pct=0)))
    text, voice, speed, lang = FakeKokoro.created[-1]
    assert lang == "en-us" and speed == pytest.approx(0.5)  # asked twice as slow...
    assert float(voice.mean()) == pytest.approx(2.0)  # 50/50 blend of the two styles
    assert clip.duration == pytest.approx(0.5, abs=0.01)  # ...and resampled back: same length, octave up
    assert KokoroTTS("bf_emma", model_path=model, voices_path=voices).lang == "en-gb"
    with pytest.raises(TTSError, match="unknown Kokoro voice"):
        KokoroTTS("xx_nobody", model_path=model, voices_path=voices)
    with pytest.raises(TTSError, match="neuroclone setup"):
        KokoroTTS("af_bella", model_path=str(tmp_path / "missing.onnx"), voices_path=voices)


@pytest.mark.skipif(not (Path(os.environ.get("NEUROCLONE_KOKORO_DIR", "models/kokoro")) / "kokoro-v1.0.onnx").exists(),
                    reason="real Kokoro model not downloaded (run `neuroclone setup`)")
def test_real_kokoro_voice():
    pytest.importorskip("kokoro_onnx")
    base = Path(os.environ.get("NEUROCLONE_KOKORO_DIR", "models/kokoro"))
    tts = KokoroTTS("af_bella", model_path=str(base / "kokoro-v1.0.onnx"), voices_path=str(base / "voices-v1.0.bin"),
                    pitch_semitones=2)
    clip = run(tts.synthesize("Hello chat!"))
    assert clip.sample_rate == 24000 and 0.5 < clip.duration < 4 and float(np.abs(clip.samples).max()) > 0.05


def test_live_priority_only_for_models_that_share_this_pc():
    from neuroclone.offline import runs_locally

    assert runs_locally(LLMConfig(provider="ollama")) and runs_locally(LLMConfig(provider="mock"))
    assert not runs_locally(LLMConfig(provider="anthropic"))
    assert not runs_locally(LLMConfig(provider="openai", base_url="https://api.example.com/v1"))
