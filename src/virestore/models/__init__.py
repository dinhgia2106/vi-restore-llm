"""Restorers share one interface: ``restore(list[str]) -> list[str]``.

Heavy back-ends (torch, transformers, llama.cpp) are imported lazily so the
baselines and the evaluation harness run without them.
"""

from __future__ import annotations

from typing import Protocol


class Restorer(Protocol):
    name: str

    def restore(self, inputs: list[str]) -> list[str]: ...


def load(kind: str, **kwargs) -> Restorer:
    if kind == "identity":
        from .baselines import Identity

        return Identity()
    if kind == "lexicon":
        from .baselines import Lexicon

        return Lexicon.from_jsonl(kwargs["train"])
    if kind == "tcn":
        from .tcn import TCNRestorer

        return TCNRestorer(kwargs["tcn_repo"], kwargs.get("tcn_weights"))
    if kind == "hf":
        from .hf_llm import HFRestorer

        return HFRestorer(kwargs["model"], adapter=kwargs.get("adapter"), batch_size=kwargs.get("batch_size", 8))
    if kind == "gguf":
        from .gguf import GGUFRestorer

        draft = None
        if kwargs.get("draft") == "lexicon":
            from .baselines import Lexicon
            from .gguf import LexiconDraft

            draft = LexiconDraft(Lexicon.from_jsonl(kwargs["train"]), max_tokens=kwargs.get("draft_tokens") or 32)
        return GGUFRestorer(kwargs["model"], n_threads=kwargs.get("threads"), draft=draft)
    if kind == "tagger":
        from .tagger import TaggerRestorer

        return TaggerRestorer(kwargs["model"], threads=kwargs.get("threads"), batch_size=kwargs.get("batch_size", 64))
    if kind == "tagger-onnx":
        from .tagger import OnnxTaggerRestorer

        return OnnxTaggerRestorer(kwargs["model"], threads=kwargs.get("threads"), batch_size=kwargs.get("batch_size", 64))
    raise ValueError(f"unknown model kind: {kind}")
