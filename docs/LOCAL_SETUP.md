# Run NeuroClone on your own PC: offline and free

This guide targets a typical streaming PC (Windows 10/11, an Intel Core i5 or similar, 16–32 GB of RAM and a dedicated GPU), but the same steps work on Linux and macOS. After a one-time download, everything runs on your machine: no API keys, no subscriptions, and no data leaves the PC. Twitch or YouTube chat still needs the internet, because that's the stream.

## What you need

- Windows 10 or 11 (64-bit), or Linux, or macOS on Apple Silicon.
- A dedicated GPU: NVIDIA (best supported), AMD Radeon or Intel Arc. It also runs without one, just slower.
- About 10 GB of free disk space for a mid-size model (up to 25 GB for the largest).
- Internet only during installation.

## Install (Windows)

1. Download or clone the repository and open the folder.
2. Double-click **`install.bat`**.

The installer:

1. finds Python 3.10–3.13, or installs Python 3.12 with `winget`;
2. installs NeuroClone into a private `.venv` folder with the local voice, microphone and vision extras;
3. installs [Ollama](https://ollama.com), the free local model server, if you don't have it, and tunes it: a half-size context cache (`OLLAMA_KV_CACHE_TYPE=q8_0`), flash attention, and room for two models at once;
4. runs `neuroclone setup --yes`, which detects your CPU, RAM and GPU, picks the best models that fit, writes `config\local.yaml`, downloads everything, and benchmarks it.

Options: `install.bat -Prefer quality -Mic -Creator "YourName"`.

Linux or macOS: `scripts/install.sh --mic --creator "YourName"` (install Ollama first from https://ollama.com/download).

## Start

- Double-click **`start.bat`** to talk to her in a terminal window. The control room is at http://127.0.0.1:8080.
- `start.bat run` goes live: Twitch/YouTube chat, voice, avatar and games, as set in `config\local.yaml`.
- `start.bat doctor` checks everything; `start.bat bench` measures reply speed.

On Linux or macOS, use `.venv/bin/neuroclone chat` or `.venv/bin/neuroclone run --console` instead.

## What setup picks for your GPU

The brain gets the GPU. The voice (Kokoro), the ears (Whisper) and the memory model run on the CPU, which has plenty of spare power while the GPU is busy thinking. While streaming, about 1.5 GB of VRAM is left for Windows, OBS and VTube Studio.

| GPU memory | speed | balanced (default) | quality |
|---|---|---|---|
| none (CPU only) | Qwen 3.5 2B | Qwen 3.5 2B | Qwen 3.5 4B (8+ cores) |
| 4 GB | Qwen 3.5 0.8B | Qwen 3.5 2B | Qwen 3.5 2B |
| 6 GB | Qwen 3.5 2B | Qwen 3.5 4B | Qwen 3.5 4B |
| 8 GB | Qwen 3.5 4B | Gemma 4 E4B | Qwen 3.5 9B (a little on the CPU) |
| 12 GB | Gemma 4 12B | Gemma 4 12B | Gemma 4 12B |
| 16 GB | Gemma 4 12B | Gemma 4 12B | Gemma 4 26B MoE (a little on the CPU) |
| 24 GB | Gemma 4 26B MoE | Gemma 4 31B | Gemma 4 31B |

With 32 GB of RAM, "quality" may place part of a model in system memory. Mixture-of-experts models (Gemma 4 26B) cope with that far better than dense ones, so the planner prefers them. Setup looks up exact model sizes in the Ollama registry before choosing, and after loading it reports how much of the model actually landed on the GPU.

Both model families can see images, so the same model also describes your screen when asked (the control room's "Look at screen" button, or `/look` in the terminal). No second model competes for VRAM.

**Streaming a game from the same PC?** The game needs VRAM too. Tell setup how much to leave for her, for example `neuroclone setup --vram 4 --force` on an 8 GB card, or use `--prefer speed`.

## Why it's fast

| Problem on a home PC | What NeuroClone does |
|---|---|
| Ollama's default context is 4k tokens on GPUs under 24 GB, silently cutting off the character prompt | Uses Ollama's native API and asks for an 8k context on every request (`llm.num_ctx`) |
| New models "think" for seconds before answering: dead air on stream | Thinking is turned off per request (`llm.think: false`) |
| Ollama unloads idle models after 5 minutes, so the next reply waits for a reload | `keep_alive: 30m`, and every model is loaded at startup (`performance.warmup`) |
| Memory summaries and reflections compete with live replies for the one GPU | Live replies go first: background work waits, and is paused and retried if a reply starts (`performance.live_priority`) |
| A second model (embeddings, vision) could evict the brain from VRAM | The memory embedder is pinned to the CPU; vision reuses the brain model |
| Voice and speech recognition would steal VRAM | Kokoro and Whisper run on the CPU, faster than real time |
| Speaking only after the whole reply is written | The first sentence (even the first clause) is spoken while the rest is still generating |
| Kokoro's int8 model is about 6× slower on CPUs (measured) | Setup downloads the full-precision model, which is faster than real time |

## Offline, verified

`config\local.yaml` sets `offline: true`. At startup NeuroClone refuses any cloud AI service (Claude, OpenAI, Azure or Edge voices, remote servers) and tells the speech libraries not to contact Hugging Face. `neuroclone doctor` shows the check. Local-network servers, for example Ollama on another PC in your house, are allowed.

## Tuning

Edit `config\local.yaml` (every key is documented in `neuroclone/config.py`), or re-run setup:

- `neuroclone setup --prefer speed|balanced|quality --force` picks again.
- `--model gemma4:12b` uses a model of your choice; `--vram 6` overrides the detected GPU memory.
- `--no-stream` is for when you aren't running OBS or VTube Studio: more VRAM for her.
- `--mic` enables the microphone (and downloads Whisper).
- Voice: `tts.voice: af_heart` (any Kokoro voice), blends such as `af_bella:0.7+af_sky:0.3`, and `tts.pitch_semitones: 2` for a brighter voice. The pitch shift has the model speak slightly slower, then resamples the result to normal speed at a higher pitch, so there are no warbly artifacts.
- `neuroclone say "[excited] Hello chat!"` tests the voice; `neuroclone bench` re-measures.

## Troubleshooting

- **"cannot reach Ollama"**: start the Ollama app from the Start menu (it lives in the system tray).
- **Slow first words, or `bench` shows less than 90% on the GPU**: the model doesn't fit. Run `neuroclone setup --prefer speed --force`, close VRAM-heavy apps, or use `--no-stream` if you aren't streaming.
- **She hears herself**: use headphones, or lower the speaker volume; she also ignores transcripts that match her own recent words.
- **Wrong speaker or microphone**: set `audio.device` or `stt.input_device` (a number from `python -m sounddevice`).
- **AMD or Intel GPU running on the CPU**: update the graphics driver. Ollama uses ROCm or Vulkan for these cards.
- **Windows SmartScreen blocks install.bat**: click "More info", then "Run anyway". The script only installs Python packages, Ollama and the model files.

## Make her yours, still free

Fine-tune a small model on your character with your own GPU and a local teacher model. See [training/README.md](../training/README.md#free-on-your-own-pc).
