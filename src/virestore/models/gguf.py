"""Quantised GGUF model through llama.cpp — the on-device target for VIMEK."""

from __future__ import annotations

import os

from ..prompts import messages
from ..text import strip_diacritics


class LexiconDraft:
    """Draft tokens for speculative decoding: the lexicon's restoration of the rest of the sentence.

    The restored sentence keeps the input's letters and changes only diacritics, and the lexicon
    gets most syllables right at almost no cost. At every step this proposes the lexicon's text
    for whatever the model has not written yet; llama.cpp checks all proposed tokens in one
    forward pass and keeps the prefix that matches its own greedy choice, so the output is exactly
    the greedy output — the draft only changes how many forward passes it takes.

    When the model's text so far no longer matches the input's letters (it rewrote a word), there
    is nothing sensible to propose and the step falls back to plain decoding.
    """

    def __init__(self, lexicon, max_tokens: int = 32):
        self.lexicon = lexicon
        self.max_tokens = max_tokens
        self.llm = None
        self.rounds = 0  # draft proposals made (= verification forward passes after the first)
        self.proposed = 0  # draft tokens proposed in total

    def attach(self, llm) -> None:
        self.llm = llm
        self.end = list(llm.tokenize(b"<|im_end|>", add_bos=False, special=True))

    def begin(self, source: str) -> None:
        self.draft = self.lexicon.restore([source])[0]
        self.base = strip_diacritics(self.draft).lower()
        self.prompt_len = None

    def __call__(self, input_ids, /, **kwargs):
        import numpy as np

        empty = np.array([], dtype=np.intc)
        if self.prompt_len is None:  # first call comes right after the first generated token
            self.prompt_len = len(input_ids) - 1
        try:
            text = self.llm.detokenize([int(t) for t in input_ids[self.prompt_len :]]).decode("utf-8")
        except UnicodeDecodeError:  # the last token ends inside a multi-byte character
            return empty
        if strip_diacritics(text).lower() != self.base[: len(text)]:
            return empty
        rest = self.draft[len(text) :]
        tokens = (list(self.llm.tokenize(rest.encode("utf-8"), add_bos=False, special=False)) if rest else []) + self.end
        tokens = tokens[: self.max_tokens]
        self.rounds += 1
        self.proposed += len(tokens)
        return np.array(tokens, dtype=np.intc)


class GGUFRestorer:
    def __init__(self, path: str, n_ctx: int = 512, n_threads: int | None = None, draft: LexiconDraft | None = None):
        from llama_cpp import Llama

        self.name = f"gguf:{os.path.basename(path)}" + ("+lexicon-draft" if draft else "")
        self.n_threads = n_threads or os.cpu_count()
        self.draft = draft
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=self.n_threads, verbose=False, draft_model=draft)
        if draft:
            draft.attach(self.llm)
        self.generated_tokens = 0

    def restore(self, inputs: list[str]) -> list[str]:
        outs = []
        for text in inputs:
            if self.draft:
                self.draft.begin(text)
            res = self.llm.create_chat_completion(messages=messages(text), temperature=0.0, max_tokens=len(text) + 32)
            self.generated_tokens += res["usage"]["completion_tokens"]
            outs.append(res["choices"][0]["message"]["content"].strip().split("\n")[0])
        return outs

    def stats(self) -> dict:
        s = {"generated_tokens": self.generated_tokens}
        if self.draft:
            s.update(draft_rounds=self.draft.rounds, draft_tokens_proposed=self.draft.proposed, draft_max_tokens=self.draft.max_tokens)
        return s

    def restore_nbest(self, inputs: list[str], k: int) -> list[list[str]]:
        raise NotImplementedError("llama-cpp-python has no beam search, so GGUF models have no n-best (EM@k) yet")
