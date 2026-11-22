"""LoRA / QLoRA supervised fine-tuning for diacritic restoration.

    python -m virestore.train_lora --config configs/qwen05_lora.yaml

Loss is computed on the assistant answer only (prompt tokens are masked with -100).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from . import utf8_stdio
from .data import load_jsonl
from .prompts import messages


class SFTDataset:
    def __init__(self, rows: list[dict], tok, max_len: int):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        prompt = self.tok.apply_chat_template(messages(r["input"]), tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(messages(r["input"], r["target"]), tokenize=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"][: self.max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids) :]
        return {"input_ids": f_ids, "labels": labels}


def collate(pad_id: int):
    import torch

    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return fn


def _warmup(training_args_cls, ratio: float) -> dict:
    """transformers 5 dropped `warmup_ratio`; its `warmup_steps` takes a float in [0, 1) as a ratio."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(training_args_cls)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in names else {"warmup_steps": ratio}


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, quantization_config=quant
    )
    if quant:
        model = prepare_model_for_kbit_training(model)
    lora = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    d = cfg["data"]
    train = SFTDataset(load_jsonl(d["train"], d.get("train_limit")), tok, d["max_len"])
    dev = SFTDataset(load_jsonl(d["dev"], d.get("dev_limit", 1000)), tok, d["max_len"])

    t = cfg["train"]
    out = Path(cfg["output_dir"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["grad_accum"],
            learning_rate=t["lr"],
            num_train_epochs=t["epochs"],
            **_warmup(TrainingArguments, t["warmup_ratio"]),
            lr_scheduler_type="cosine",
            logging_steps=t["logging_steps"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_steps=t["eval_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=t.get("gradient_checkpointing", False),
            report_to=t.get("report_to", "none"),
            seed=cfg.get("seed", 13),
        ),
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collate(tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
