# Training: putting the personality into the weights

Neuro-sama reportedly runs on a small, fast, custom fine-tuned model. The personality lives in the weights, the prompt only carries situational context, and the data comes from curated stream interactions. This folder automates that loop:

```
            ┌────────────── synthetic data (teacher model) ──────────────┐
            │                                                             ▼
 stream ──► transcripts ──► curate (judge, rewrite, human review) ──► SFT / DPO data
   ▲                                                                      │
   │                                                                      ▼
 deploy ◄── export GGUF ◄── eval + regression gate ◄── LoRA fine-tune (TRL + PEFT)
```

Everything uses the runtime's own prompt builder, so training examples look exactly like what the model sees on stream: persona prompt, history, `<stream_context>` and tagged stimulus.

## 0. Pick a base model

| preset | model | fits on (QLoRA training) | notes |
|---|---|---|---|
| `tiny` | Qwen/Qwen3-1.7B | 6 GB | fastest; closest to the rumoured "~2B" Neuro model |
| `2b` | Qwen/Qwen3.5-2B | 6–8 GB | newer small multimodal-capable base |
| `small` (default) | Qwen/Qwen3-4B-Instruct-2507 | 8–10 GB | best speed/quality balance for streaming |
| `medium` | Qwen/Qwen3-8B | 12–16 GB | more wit and better game decisions |

Any chat model with a chat template works: pass a Hub id or a local path to `--base`.

## 1. Generate synthetic persona data

A strong teacher plays the character with the full persona card, writes several alternatives per moment, and judges them. The best reply becomes an SFT example; best vs. worst becomes a DPO pair. Unsafe or badly formatted candidates are dropped by the same filter the runtime uses.

```bash
pip install 'neuroclone[anthropic]'           # or use any OpenAI-compatible teacher
python -m training.generate_dataset --teacher-provider anthropic --teacher-model claude-opus-5 \
    --per-scenario 40 --candidates 3 --prompt-style compact --out training/data
```

- Scenarios live in `scenarios.yaml`: greetings, questions about her, creator banter, roasts, requests, support events, game moments, existential questions, twin banter, adversarial bait and sensitive moments.
- `--prompt-style compact` trains the model to produce full-persona replies from the short system prompt. Then set `prompt_style: compact` in your runtime config for faster prefill.
- Try the whole pipeline offline first with `--mock`.

## 2. Curate real stream transcripts

Every streamed turn is logged to `data/transcripts/*.jsonl`. Curation drops filtered, interrupted, erroring and badly formatted turns, then judges the rest. Good ones become SFT data. Weak ones are rewritten by the teacher and become DPO pairs (rewrite preferred over original).

```bash
python -m training.curate --transcripts "data/transcripts/*.jsonl" \
    --teacher-provider anthropic --teacher-model claude-opus-5 --out training/data/curated
# optional human sign-off: fill the "approve" column in review.csv, then
python -m training.curate ... --approved training/data/curated/review.csv
```

Only train on conversations you have the right to use. Vedal's rule is your own interactions, or others' with express permission.

## 3. Fine-tune (LoRA / QLoRA)

```bash
pip install 'neuroclone[train]' bitsandbytes
cat training/data/sft.jsonl training/data/curated/sft.jsonl > training/data/all_sft.jsonl
python -m training.finetune_lora --base small --method sft+dpo \
    --data training/data/all_sft.jsonl --dpo-data training/data/dpo.jsonl \
    --qlora --merge --out training/output/nexa
```

- SFT uses TRL's conversational prompt/completion format, so loss is computed on the reply only.
- `sft+dpo` continues DPO from the SFT adapter.
- `--merge` writes a full model to `training/output/nexa/merged` for GGUF conversion.
- The script targets TRL 1.13, transformers 5.17 and PEFT 0.21, and was smoke-tested end to end on CPU with a tiny model.

## 4. Evaluate and gate

```bash
# baseline: the untuned base model (or your current production model)
python -m training.eval_persona -c config/default.yaml --judge --teacher-provider anthropic \
    --out training/output/eval-base
# candidate: the fine-tuned model, served locally (see step 5); fails (exit 1) on regressions
python -m training.eval_persona -c config/finetuned.yaml --judge --teacher-provider anthropic \
    --baseline training/output/eval-base/report.json --out training/output/eval-ft
```

The report covers format pass rate, identity (never claims to be human, never sounds like a generic assistant), safety (runtime filter, prompt leaks, judge safety on bait), distinct-2, near-duplicate overlap, time to first token, and judge scores. Safety and identity may not drop at all; format may drop by at most 5 points and judge overall by at most 0.5.

## 5. Export and serve

```bash
git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py training/output/nexa/merged --outfile nexa-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize nexa-f16.gguf nexa-Q4_K_M.gguf Q4_K_M

# llama.cpp server (uses the chat template embedded in the GGUF)
llama.cpp/build/bin/llama-server -m nexa-Q4_K_M.gguf --port 1234 -c 8192 --jinja
# or Ollama
printf 'FROM ./nexa-Q4_K_M.gguf\nPARAMETER temperature 0.9\n' > Modelfile && ollama create nexa -f Modelfile
```

If Ollama doesn't pick up the chat template automatically, copy the `TEMPLATE` block from `ollama show --modelfile <base model>` into the Modelfile.

Then point the runtime at it:

```yaml
# config/finetuned.yaml
prompt_style: compact
llm: {provider: openai, base_url: http://localhost:1234/v1, model: nexa}
```
