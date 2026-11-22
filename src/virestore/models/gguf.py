"""Quantised GGUF model through llama.cpp — the on-device target for VIMEK."""

from __future__ import annotations

import os

from ..prompts import messages


class GGUFRestorer:
    def __init__(self, path: str, n_ctx: int = 512, n_threads: int | None = None):
        from llama_cpp import Llama

        self.name = f"gguf:{os.path.basename(path)}"
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(), verbose=False)

    def restore(self, inputs: list[str]) -> list[str]:
        outs = []
        for text in inputs:
            res = self.llm.create_chat_completion(messages=messages(text), temperature=0.0, max_tokens=len(text) + 32)
            outs.append(res["choices"][0]["message"]["content"].strip().split("\n")[0])
        return outs
