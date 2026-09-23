from neuroclone.cli import build_parser, main


def test_parser_commands():
    parser = build_parser()
    for argv in (["run", "--mock"], ["chat", "--no-idle"], ["neuro-api", "--port", "9000"], ["doctor"],
                 ["say", "hello"], ["persona", "list"], ["memory", "stats"], ["blocklist", "fetch"]):
        assert parser.parse_args(argv).func


def test_persona_commands(capsys):
    assert main(["persona", "list"]) == 0
    assert "nexa" in capsys.readouterr().out
    assert main(["persona", "show", "vexa", "--mock"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("You are Vexa") and "Hard rules" in out


def test_doctor_in_mock_mode(capsys, tmp_path):
    code = main(["doctor", "--mock", "--set", f"memory.path={tmp_path / 'm.sqlite3'}",
                 "--set", "games.port=0", "--set", "overlay.port=0"])
    out = capsys.readouterr().out
    assert code == 0 and "all good" in out and "mock" in out


def test_memory_stats_and_config_errors(capsys, tmp_path):
    assert main(["memory", "stats", "--set", f"memory.path={tmp_path / 'm.sqlite3'}"]) == 0
    assert '"memories": 0' in capsys.readouterr().out
    assert main(["doctor", "--set", "llm.bogus=1"]) == 1  # doctor reports the problem itself
    assert main(["persona", "show", "--set", "llm.bogus=1"]) == 2  # other commands exit with a config error
    assert "unknown key" in capsys.readouterr().err
