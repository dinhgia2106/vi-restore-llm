"""Character-level diacritic tagger (see virestore.tagger) as a restorer."""

from __future__ import annotations

from pathlib import Path


class TaggerRestorer:
    def __init__(self, path: str, threads: int | None = None, batch_size: int = 64):
        import torch

        from ..tagger import load

        if threads:
            torch.set_num_threads(threads)
        self.device = "cuda" if torch.cuda.is_available() and not threads else "cpu"
        self.model, self.vocab, self.cfg = load(Path(path), self.device)
        self.max_len = self.cfg["model"]["max_len"]
        self.batch_size = batch_size
        self.n_threads = torch.get_num_threads() if self.device == "cpu" else None
        self.name = f"tagger:{Path(path).name}"

    def restore(self, inputs: list[str]) -> list[str]:
        import torch

        from ..tagger import decode

        out: list[str] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            n = min(max(len(s) for s in batch), self.max_len)
            ids = torch.zeros((len(batch), n), dtype=torch.long)
            for k, s in enumerate(batch):
                enc = self.vocab.encode(s[:n])
                ids[k, : len(enc)] = torch.tensor(enc)
            ids = ids.to(self.device)
            with torch.inference_mode():
                labels = self.model(ids, ids != 0).argmax(-1).tolist()
            # characters beyond max_len (very long sentences) are copied unchanged
            out += [decode(s[:n], lab[: min(len(s), n)]) + s[n:] for s, lab in zip(batch, labels)]
        return out
