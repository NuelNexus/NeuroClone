"""LoRA / QLoRA fine-tuning of a small open model on NeuroClone persona data.

Mirrors the approach attributed to Neuro-sama (a small, fast, fine-tuned local model with the
personality in its weights), automated:  SFT on the best teacher/curated replies, then optional
DPO on best-vs-worst pairs.

  pip install 'neuroclone[train]'        (plus bitsandbytes for --qlora on NVIDIA GPUs)
  python -m training.finetune_lora --base small --data training/data/sft.jsonl --out training/output/nexa
  python -m training.finetune_lora --base small --method sft+dpo --data training/data/sft.jsonl \\
      --dpo-data training/data/dpo.jsonl --qlora --merge

Datasets use TRL's conversational formats:
  SFT: {"prompt": [messages...], "completion": [{"role": "assistant", ...}]}  (loss on the reply only)
  DPO: {"prompt": [...], "chosen": [...], "rejected": [...]}
Written against TRL 1.13 / transformers 5.17 / PEFT 0.21.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PRESETS = {
    "tiny": "Qwen/Qwen3-1.7B",
    "2b": "Qwen/Qwen3.5-2B",
    "small": "Qwen/Qwen3-4B-Instruct-2507",
    "medium": "Qwen/Qwen3-8B",
}


def resolve_base(name: str) -> str:
    return PRESETS.get(name, name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="small", help=f"preset {sorted(PRESETS)}, a Hub id, or a local path")
    p.add_argument("--data", default="training/data/sft.jsonl")
    p.add_argument("--dpo-data", default="training/data/dpo.jsonl")
    p.add_argument("--method", choices=["sft", "dpo", "sft+dpo"], default="sft")
    p.add_argument("--out", default="training/output/persona-lora")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="overrides epochs when > 0")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dpo-lr", type=float, default=5e-6)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-split", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true", help="4-bit base weights (needs bitsandbytes + CUDA)")
    p.add_argument("--cpu", action="store_true", help="force CPU (smoke tests only)")
    p.add_argument("--merge", action="store_true", help="also save a merged full model for GGUF conversion")
    p.add_argument("--seed", type=int, default=42)
    return p


def _dtype_and_bf16(args):
    import torch

    if args.cpu or not torch.cuda.is_available():
        return torch.float32, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    return torch.float16, False


def _quantization(args, dtype):
    if not args.qlora:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=dtype)


def _lora(args):
    from peft import LoraConfig

    return LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")


def load_split(path: str, keys: tuple[str, ...], eval_split: float, seed: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")
    missing = [k for k in keys if k not in ds.column_names]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; expected keys {keys}")
    ds = ds.remove_columns([c for c in ds.column_names if c not in keys])
    if eval_split > 0 and len(ds) >= 20:
        split = ds.train_test_split(test_size=eval_split, seed=seed)
        return split["train"], split["test"]
    return ds, None


def _check_not_empty(trainer, raw_count: int, max_len: int) -> None:
    kept = len(trainer.train_dataset)
    if kept == 0:
        raise SystemExit(f"all {raw_count} examples were dropped during preparation; they are probably longer than "
                         f"--max-len {max_len} tokens (try a larger --max-len or --prompt-style compact data)")
    if kept < raw_count:
        print(f"note: {raw_count - kept} of {raw_count} examples were dropped (too long for --max-len {max_len})")


def load_tokenizer(base: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{base} has no chat template; pick an instruct/chat model")
    return tok


def train_sft(args, base: str, out: Path) -> Path:
    from trl import SFTConfig, SFTTrainer

    dtype, bf16 = _dtype_and_bf16(args)
    train, evaluation = load_split(args.data, ("prompt", "completion"), args.eval_split, args.seed)
    tokenizer = load_tokenizer(base)
    cfg = SFTConfig(
        output_dir=str(out / "sft-checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float < 1 is a ratio of total steps (transformers 5)
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if evaluation is not None else "no",
        bf16=bf16,
        max_length=args.max_len,
        report_to="none",
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu,
        model_init_kwargs={"dtype": dtype},
        seed=args.seed,
    )
    trainer = SFTTrainer(model=base, args=cfg, train_dataset=train, eval_dataset=evaluation,
                         processing_class=tokenizer, peft_config=_lora(args),
                         quantization_config=_quantization(args, dtype))
    _check_not_empty(trainer, len(train), args.max_len)
    result = trainer.train()
    adapter = out / "sft"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (adapter / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    return adapter


def train_dpo(args, base: str, out: Path, init_adapter: Optional[Path]) -> Path:
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    dtype, bf16 = _dtype_and_bf16(args)
    train, evaluation = load_split(args.dpo_data, ("prompt", "chosen", "rejected"), args.eval_split, args.seed)
    tokenizer = load_tokenizer(base)
    cfg = DPOConfig(
        output_dir=str(out / "dpo-checkpoints"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=max(1, args.batch // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.dpo_lr,
        beta=args.dpo_beta,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float < 1 is a ratio of total steps (transformers 5)
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if evaluation is not None else "no",
        bf16=bf16,
        max_length=args.max_len,
        report_to="none",
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu,
        model_init_kwargs={"dtype": dtype} if init_adapter is None else None,
        seed=args.seed,
    )
    if init_adapter is not None:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, quantization_config=_quantization(args, dtype))
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=train, eval_dataset=evaluation,
                             processing_class=tokenizer)
    else:
        trainer = DPOTrainer(model=base, ref_model=None, args=cfg, train_dataset=train, eval_dataset=evaluation,
                             processing_class=tokenizer, peft_config=_lora(args),
                             quantization_config=_quantization(args, dtype))
    _check_not_empty(trainer, len(train), args.max_len)
    result = trainer.train()
    adapter = out / "dpo"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (adapter / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    return adapter


def merge(base: str, adapter: Path, out: Path) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype="auto")
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    target = out / "merged"
    merged.save_pretrained(str(target))
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(target))
    return target


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base = resolve_base(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter: Optional[Path] = None
    if args.method in ("sft", "sft+dpo"):
        adapter = train_sft(args, base, out)
        print(f"SFT adapter saved to {adapter}")
    if args.method in ("dpo", "sft+dpo"):
        adapter = train_dpo(args, base, out, adapter)
        print(f"DPO adapter saved to {adapter}")
    if args.merge and adapter is not None:
        merged = merge(base, adapter, out)
        print(f"merged model saved to {merged}")
        print("next: convert to GGUF with llama.cpp and serve it (see training/README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
