#!/usr/bin/env bash
# NeuroClone installer for Linux and macOS: packages into .venv, then `neuroclone setup`
# (detects the GPU, picks models, writes config/local.yaml, downloads everything once).
#   scripts/install.sh [--prefer speed|balanced|quality] [--mic] [--creator "Your Name"]
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
fi
[ -n "$PY" ] || { echo "Python 3.10+ is required"; exit 1; }

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -e '.[local]'

if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
  echo "note: audio playback needs PortAudio: sudo apt install libportaudio2 (or your distro's equivalent)"
fi

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama (the local LLM server) is not installed. Install it from https://ollama.com/download"
    echo "(Linux: curl -fsSL https://ollama.com/install.sh | sh), start it, then run this script again."
    exit 1
  fi
  echo "starting Ollama..."
  (OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=2 nohup ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
fi

.venv/bin/neuroclone setup --yes "$@"
echo
echo "Start chatting: .venv/bin/neuroclone chat      Go live: .venv/bin/neuroclone run --console"
