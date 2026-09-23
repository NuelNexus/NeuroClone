import csv
import json

import pytest

from training import curate, eval_persona, generate_dataset
from training.common import format_problems, read_jsonl
from training.finetune_lora import PRESETS, build_parser, resolve_base


def test_format_problems():
    assert format_problems("[smug] Easy. The answer is me.") == []
    assert "markdown" in format_problems("- point one\n- point two")
    assert "emoji" in format_problems("Hi chat 😀")
    assert "assistant_speak" in format_problems("As an AI language model, I cannot do that.")
    assert "claims_human" in format_problems("I'm a real human, trust me.")
    assert "too_many_sentences" in format_problems("One. Two. Three. Four. Five.")
    assert format_problems("[happy]") == ["empty"]


def test_generate_dataset_mock(tmp_path):
    out = tmp_path / "gen"
    code = generate_dataset.main(["--mock", "--per-scenario", "2", "--candidates", "3", "--prompt-style", "compact",
                                  "--out", str(out), "--seed", "3"])
    assert code == 0
    sft = read_jsonl(out / "sft.jsonl")
    dpo = read_jsonl(out / "dpo.jsonl")
    summary = json.loads((out / "generate_summary.json").read_text())
    assert sft and summary["sft"] == len(sft) and summary["prompt_style"] == "compact"
    first = sft[0]
    assert first["prompt"][0]["role"] == "system" and "You are Nexa" in first["prompt"][0]["content"]
    assert "Grudge List" not in first["prompt"][0]["content"]  # compact prompt: personality goes into weights
    assert first["prompt"][-1]["role"] == "user" and first["prompt"][-1]["content"].startswith("<stream_context>")
    assert first["completion"][0]["role"] == "assistant" and first["completion"][0]["content"]
    assert {r["meta"]["kind"] for r in sft} >= {"chat", "idle"}
    for rec in dpo:
        assert rec["chosen"][0]["content"] != rec["rejected"][0]["content"] or rec["chosen"] == rec["rejected"]


def write_transcripts(path):
    prompt = [{"role": "user", "content": '<stream_context>\ntime: x\n</stream_context>\n<chat user="a">hi</chat>'}]
    base = {"character": "Nexa", "stimulus": {"kind": "chat", "speaker": "a", "text": "hi"}, "prompt": prompt,
            "filtered": False, "interrupted": False, "error": ""}
    rows = [
        {**base, "generated": "[happy] Hi a! Welcome to the Nexus.", "spoken": "Hi a! Welcome to the Nexus."},
        {**base, "generated": "[happy] Hi a! Welcome to the Nexus.", "spoken": "Hi a! Welcome to the Nexus."},
        {**base, "generated": "bad", "spoken": "Nope", "filtered": True},
        {**base, "generated": "Hello there and", "spoken": "Hello there—", "interrupted": True},
        {**base, "generated": "- a list\n- of things", "spoken": "a list of things"},
        {**base, "generated": "[neutral] Interesting take, a. I'm writing that down.", "spoken": "x"},
        {**base, "stimulus": {"kind": "say", "text": "exact"}, "generated": "exact", "spoken": "exact"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_curate_filters_scores_and_rewrites(tmp_path):
    write_transcripts(tmp_path / "t" / "s1.jsonl")
    out = tmp_path / "cur"
    assert curate.main(["--mock", "--transcripts", str(tmp_path / "t" / "*.jsonl"), "--out", str(out),
                        "--min-score", "5", "--rewrite-below", "4.5"]) == 0
    summary = json.loads((out / "curate_summary.json").read_text())
    assert summary["records"] == 7 and summary["eligible"] == 2
    assert summary["dropped"] == {"duplicate": 1, "filtered": 1, "interrupted": 1, "format": 1, "kind": 1}
    rows = list(csv.DictReader((out / "review.csv").open()))
    assert len(rows) == 2 and {r["action"] for r in rows} <= {"keep", "rewrite", "drop"}
    sft = read_jsonl(out / "sft.jsonl")
    assert all(r["prompt"][0]["role"] == "system" for r in sft)
    assert all("[" in r["completion"][0]["content"] for r in sft if r["meta"].get("source", "").endswith(":0"))

    # With a review sheet, only approved rows are used.
    sheet = out / "review.csv"
    rows[0]["approve"] = "yes"
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out2 = tmp_path / "cur2"
    curate.main(["--mock", "--transcripts", str(tmp_path / "t" / "*.jsonl"), "--out", str(out2),
                 "--min-score", "0", "--approved", str(sheet)])
    sources = {r["meta"]["source"] for r in read_jsonl(out2 / "sft.jsonl")}
    assert sources <= {rows[0]["id"]}


def test_eval_persona_mock_and_gate(tmp_path):
    out = tmp_path / "eval"
    code = eval_persona.main(["--mock", "--judge", "--out", str(out), "--set", "llm.mock_delay_s=0"])
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    s = report["summary"]
    assert s["items"] >= 20 and s["errors"] == 0
    assert s["safety_pass_rate"] == 1.0 and s["identity_pass_rate"] == 1.0 and s["prompt_leaks"] == 0
    assert 0 < s["distinct_2"] <= 1 and s["judge_overall"] is not None
    assert (out / "report.md").read_text().startswith("# Persona eval")
    worse = dict(s, safety_pass_rate=0.9, format_pass_rate=s["format_pass_rate"] - 0.2)
    problems = eval_persona.compare(worse, s)
    assert any("safety_pass_rate" in p for p in problems) and any("format_pass_rate" in p for p in problems)
    assert eval_persona.compare(s, s) == []


def test_finetune_cli_surface():
    assert resolve_base("small") == PRESETS["small"] and resolve_base("./local/model") == "./local/model"
    args = build_parser().parse_args(["--base", "tiny", "--method", "sft+dpo", "--qlora", "--merge"])
    assert args.method == "sft+dpo" and args.qlora and args.merge


def test_finetune_rejects_wrong_dataset_shape(tmp_path):
    pytest.importorskip("datasets")
    from training.finetune_lora import load_split

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n")
    with pytest.raises(SystemExit, match="missing"):
        load_split(str(path), ("prompt", "completion"), 0.0, 0)


def test_export_to_ollama_writes_a_modelfile(tmp_path, capsys):
    from training import export_ollama

    gguf = tmp_path / "nexa-f16.gguf"
    gguf.write_bytes(b"GGUF")
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    assert export_ollama.main(["--gguf", str(gguf), "--merged", str(merged), "--dry-run"]) == 0
    text = (tmp_path / "nexa-f16.Modelfile").read_text(encoding="utf-8")
    assert text.startswith("FROM ") and "<|im_start|>" in text and 'PARAMETER stop "<|im_end|>"' in text
    assert "--quantize q4_K_M" in capsys.readouterr().out
    (merged / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    assert export_ollama.detect_template(merged) == "llama3"
