"""Turn a fine-tuned GGUF into an Ollama model the runtime can use offline.

  python llama.cpp/convert_hf_to_gguf.py training/output/nexa/merged --outfile nexa-f16.gguf --outtype f16
  python -m training.export_ollama --gguf nexa-f16.gguf --name nexa --merged training/output/nexa/merged

Ollama quantises the f16 file itself (``--quantize q4_K_M``), so no llama.cpp build is needed, which
keeps this working on Windows. Then set ``llm.model: nexa`` and ``prompt_style: compact``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Chat templates in Ollama's Go-template syntax. They render every message (system included) so the
# fine-tuned model sees exactly the layout it was trained on.
TEMPLATES = {
    "chatml": (  # Qwen, and most models fine-tuned with ChatML
        '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n',
        ["<|im_end|>", "<|im_start|>"],
    ),
    "llama3": (
        '{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>'
        '{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n',
        ["<|eot_id|>", "<|start_header_id|>"],
    ),
    "gemma": (  # Gemma has no system role: system text becomes a user turn
        '{{- range .Messages }}<start_of_turn>{{ if eq .Role "assistant" }}model{{ else }}user{{ end }}\n'
        '{{ .Content }}<end_of_turn>\n{{ end }}<start_of_turn>model\n',
        ["<end_of_turn>", "<start_of_turn>"],
    ),
}


def detect_template(merged: Optional[Path]) -> str:
    """Pick the chat format from the merged model's config.json (defaults to ChatML)."""
    if merged is None:
        return "chatml"
    try:
        config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "chatml"
    kind = str(config.get("model_type", "")).lower()
    if kind.startswith("llama"):
        return "llama3"
    if kind.startswith("gemma"):
        return "gemma"
    return "chatml"


def modelfile(gguf: Path, template: str, *, num_ctx: int = 8192, temperature: float = 0.9) -> str:
    text, stops = TEMPLATES[template]
    lines = [f"FROM {gguf.resolve().as_posix()}", f'TEMPLATE """{text}"""']
    lines += [f'PARAMETER stop "{s}"' for s in stops]
    lines += [f"PARAMETER num_ctx {num_ctx}", f"PARAMETER temperature {temperature}"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, help="the converted model (f16 or already quantised)")
    p.add_argument("--name", default="nexa", help="the Ollama model name to create")
    p.add_argument("--merged", default=None, help="merged HF model dir, to detect the chat format")
    p.add_argument("--template", choices=sorted(TEMPLATES), default=None, help="override the chat format")
    p.add_argument("--quantize", default="q4_K_M", help="Ollama quantisation for f16 input ('' to keep as is)")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--dry-run", action="store_true", help="write the Modelfile but don't call ollama")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gguf = Path(args.gguf)
    if not gguf.exists():
        raise SystemExit(f"{gguf} not found")
    template = args.template or detect_template(Path(args.merged) if args.merged else None)
    path = gguf.with_suffix(".Modelfile")
    path.write_text(modelfile(gguf, template, num_ctx=args.num_ctx), encoding="utf-8")
    print(f"wrote {path} ({template} chat format)")
    cmd = ["ollama", "create", args.name, "-f", str(path)]
    if args.quantize and any(kind in gguf.name.lower() for kind in ("f16", "f32")):  # f16 and bf16 alike
        cmd[3:3] = ["--quantize", args.quantize]
    if args.dry_run:
        print("would run: " + " ".join(cmd))
        return 0
    if shutil.which("ollama") is None:
        print("ollama is not on PATH; run this yourself:\n  " + " ".join(cmd))
        return 1
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode
    print(f"\ncreated Ollama model {args.name!r}. In your config:\n"
          f"  prompt_style: compact\n  llm: {{provider: ollama, model: {args.name}}}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
