import pytest

from neuroclone.config import Config, ConfigError, apply_mock_profile, expand_env, load_config, parse_override


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.persona == "nexa"
    assert cfg.games.port == 8000  # Neuro SDK default
    assert cfg.overlay.host == "127.0.0.1"  # dashboard never exposed by default


def test_yaml_env_and_coercion(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_PORT", "9001")
    monkeypatch.setenv("NC_MODEL", "qwen3:8b")
    path = tmp_path / "c.yaml"
    path.write_text(
        "llm:\n  model: ${NC_MODEL}\n  base_url: ${MISSING_VAR:-http://x:1/v1}\n"
        "games:\n  port: ${NC_PORT}\n  voluntary_actions: 'false'\n"
        "vision:\n  llm: {provider: mock, model: v}\n"
    )
    cfg = load_config(path)
    assert cfg.llm.model == "qwen3:8b"
    assert cfg.llm.base_url == "http://x:1/v1"
    assert cfg.games.port == 9001 and isinstance(cfg.games.port, int)
    assert cfg.games.voluntary_actions is False
    assert cfg.vision.llm.provider == "mock"


def test_unknown_keys_are_errors(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("llm:\n  modle: typo\n")
    with pytest.raises(ConfigError, match="modle"):
        load_config(path)


def test_type_errors_are_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("games:\n  port: lots\n")
    with pytest.raises(ConfigError, match="games.port"):
        load_config(path)


def test_overrides_merge_over_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("llm:\n  model: a\n  temperature: 0.5\n")
    cfg = load_config(path, ["llm.model=b", "conductor.max_sentences=2"])
    assert cfg.llm.model == "b" and cfg.llm.temperature == 0.5
    assert cfg.conductor.max_sentences == 2


def test_parse_override_and_env():
    assert parse_override("a.b.c=1") == {"a": {"b": {"c": 1}}}
    with pytest.raises(ConfigError):
        parse_override("nonsense")
    assert expand_env({"x": ["${NOPE:-d}"]}) == {"x": ["d"]}


def test_mock_profile_is_offline():
    cfg = apply_mock_profile(Config())
    assert cfg.llm.provider == "mock" and cfg.tts.provider == "silent" and cfg.audio.player == "null"
    assert not cfg.stt.enabled and not cfg.avatar.enabled


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config"
    for path in [root / "default.yaml", *sorted((root / "examples").glob("*.yaml"))]:
        load_config(path)
