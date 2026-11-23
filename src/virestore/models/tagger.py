"""Character-level diacritic tagger (see virestore.tagger) as a restorer."""

from __future__ import annotations

import heapq
import re
from pathlib import Path


def resolve_stacked_tones(source: str, labels: list[int], log_probs, valid) -> list[int]:
    """Keep one tone per run of adjacent vowels: the choice with the best total log-probability."""
    from ..tagger import TONES

    labels = list(labels)
    for m in re.finditer(r"[aeiouy]+", source, re.IGNORECASE):
        toned = [j for j in range(m.start(), m.end()) if labels[j] % len(TONES)]
        if len(toned) < 2:
            continue
        untoned = {
            j: max((l for l in valid(source[j]) if l % len(TONES) == 0), key=lambda l: float(log_probs[j, l]))
            for j in toned
        }
        keep = max(
            toned,
            key=lambda t: float(log_probs[t, labels[t]]) + sum(float(log_probs[j, untoned[j]]) for j in toned if j != t),
        )
        for j in toned:
            if j != keep:
                labels[j] = untoned[j]
    return labels


class TaggerRestorer:
    def __init__(self, path: str, threads: int | None = None, batch_size: int = 64):
        import torch

        from ..tagger import N_LABELS, load, valid_labels

        if threads:
            torch.set_num_threads(threads)
        self.device = "cuda" if torch.cuda.is_available() and not threads else "cpu"
        self.model, self.vocab, self.cfg = load(Path(path), self.device)
        self.max_len = self.cfg["model"]["max_len"]
        self.batch_size = batch_size
        self.n_threads = torch.get_num_threads() if self.device == "cpu" else None
        self.name = f"tagger:{Path(path).name}"
        # per input character, which labels give a real letter; others are masked before argmax
        self._mask_cache: dict[str, object] = {}
        self._n_labels, self._valid = N_LABELS, valid_labels

    def _mask(self, ch: str):
        import torch

        m = self._mask_cache.get(ch)
        if m is None:
            m = torch.full((self._n_labels,), float("-inf"))
            m[self._valid(ch)] = 0.0
            self._mask_cache[ch] = m
        return m

    def _log_probs(self, batch: list[str]):
        """Per-character log-probabilities over the valid labels, for a batch of sentences."""
        import torch

        n = min(max(len(s) for s in batch), self.max_len)
        if n == 0:  # all sentences empty: onnxruntime cannot run a zero-length sequence
            return torch.zeros((len(batch), 0, self._n_labels)), 0
        ids = torch.zeros((len(batch), n), dtype=torch.long)
        mask = torch.zeros((len(batch), n, self._n_labels))
        for k, s in enumerate(batch):
            s = s[:n]
            ids[k, : len(s)] = torch.tensor(self.vocab.encode(s))
            if s:
                mask[k, : len(s)] = torch.stack([self._mask(c) for c in s])
        ids = ids.to(self.device)
        with torch.inference_mode():
            logits = self.model(ids, ids != 0).float().cpu()
        return torch.log_softmax(logits + mask, dim=-1), n

    def restore(self, inputs: list[str]) -> list[str]:
        from ..tagger import decode

        out: list[str] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            lp, n = self._log_probs(batch)
            labels = lp.argmax(-1).tolist()
            # characters beyond max_len (very long sentences) are copied unchanged
            out += [decode(s[:n], lab[: min(len(s), n)]) + s[n:] for s, lab in zip(batch, labels)]
        return out

    def restore_one_tone(self, inputs: list[str]) -> list[str]:
        """Like restore(), but a run of adjacent vowels never gets two tones (used by virestore.ime)."""
        from ..tagger import decode

        out: list[str] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            lp, n = self._log_probs(batch)
            labels = lp.argmax(-1).tolist()
            for b, s in enumerate(batch):
                lab = resolve_stacked_tones(s[:n], labels[b][: min(len(s), n)], lp[b], self._valid)
                out.append(decode(s[:n], lab) + s[n:])
        return out

    def restore_nbest(self, inputs: list[str], k: int, per_char: int = 3) -> list[list[str]]:
        return self._nbest(inputs, k, per_char)

    def _nbest(self, inputs: list[str], k: int, per_char: int) -> list[list[str]]:
        """Top-k sentences by beam search over the per-character label distributions."""
        from ..tagger import TAGGABLE, apply_label

        beam = max(4 * k, 16)
        results: list[list[str]] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            lp, n = self._log_probs(batch)
            for b, s in enumerate(batch):
                hyps = [(0.0, "")]
                for j, c in enumerate(s[:n]):
                    if c not in TAGGABLE:
                        hyps = [(score, text + c) for score, text in hyps]
                        continue
                    top = lp[b, j].topk(min(per_char, len(self._valid(c))))
                    opts = [(float(v), apply_label(c, int(l))) for v, l in zip(top.values, top.indices)]
                    hyps = heapq.nlargest(beam, ((score + v, text + ch) for score, text in hyps for v, ch in opts))
                tail = s[n:]
                results.append(list(dict.fromkeys(text + tail for _, text in hyps))[:k])
        return results


class OnnxTaggerRestorer(TaggerRestorer):
    """The same tagger exported with virestore.export_onnx, run by onnxruntime (no PyTorch model)."""

    def __init__(self, path: str, threads: int | None = None, batch_size: int = 64, onnx_file: str = "tagger.onnx"):
        import json

        import onnxruntime as ort

        from ..tagger import N_LABELS, Vocab, valid_labels

        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        self.vocab = Vocab(self.cfg["vocab"])
        self.max_len = self.cfg["model"]["max_len"]
        self.batch_size = batch_size
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(p / onnx_file), opts, providers=["CPUExecutionProvider"])
        self.device, self.n_threads = "cpu", threads
        self.name = f"tagger-onnx:{p.name}/{onnx_file}"
        self._mask_cache = {}
        self._n_labels, self._valid = N_LABELS, valid_labels

    def _log_probs(self, batch: list[str]):
        import numpy as np
        import torch

        n = min(max(len(s) for s in batch), self.max_len)
        if n == 0:  # all sentences empty: onnxruntime cannot run a zero-length sequence
            return torch.zeros((len(batch), 0, self._n_labels)), 0
        ids = np.zeros((len(batch), n), dtype=np.int64)
        mask = torch.zeros((len(batch), n, self._n_labels))
        for k, s in enumerate(batch):
            s = s[:n]
            ids[k, : len(s)] = self.vocab.encode(s)
            if s:
                mask[k, : len(s)] = torch.stack([self._mask(c) for c in s])
        logits = torch.from_numpy(self.sess.run(["logits"], {"ids": ids})[0]).float()
        return torch.log_softmax(logits + mask, dim=-1), n
