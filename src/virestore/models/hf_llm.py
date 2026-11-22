"""Hugging Face causal LM restorer (base model, or base + LoRA adapter)."""

from __future__ import annotations

from ..prompts import messages


class HFRestorer:
    def __init__(self, model_id: str, adapter: str | None = None, batch_size: int = 8, max_new_tokens: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}" + (f"+{adapter}" if adapter else "")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(adapter or model_id)
        self.tok.padding_side = "left"  # required for batched generation with decoder-only models
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def _prompt(self, text: str) -> str:
        return self.tok.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)

    def restore(self, inputs: list[str]) -> list[str]:
        import torch

        outputs: list[str] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            enc = self.tok([self._prompt(t) for t in batch], return_tensors="pt", padding=True).to(self.device)
            # Output is about as long as the input; accented text usually needs more tokens, so allow 2x.
            src_tokens = max(len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in batch)
            limit = self.max_new_tokens or src_tokens * 2 + 16
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=limit, do_sample=False, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1] :]
            outputs += [t.strip().split("\n")[0] for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        return outputs
