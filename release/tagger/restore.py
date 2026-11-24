"""Restore Vietnamese diacritics with the int8 ONNX character tagger.

Needs only numpy and onnxruntime:

    pip install numpy onnxruntime
    python restore.py "hom nay troi dep qua"

The model labels every vowel (and "d") with one of 30 labels = modifier (none, breve,
circumflex, horn, stroke) x tone (none, grave, acute, hook, tilde, dot below). Other characters
are copied, so the output always keeps the input's letters.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

TONES = ("", "̀", "́", "̉", "̃", "̣")
MODIFIERS = ("", "̆", "̂", "̛", "stroke")
ALLOWED = {"a": (0, 1, 2), "e": (0, 2), "i": (0,), "o": (0, 2, 3), "u": (0, 3), "y": (0,), "d": (0, 4)}
N_LABELS = len(MODIFIERS) * len(TONES)


def valid_labels(ch: str) -> list[int]:
    low = ch.lower()
    if low not in ALLOWED:
        return [0]
    if low == "d":
        return [0, 4 * len(TONES)]
    return [m * len(TONES) + t for m in ALLOWED[low] for t in range(len(TONES))]


def apply_label(ch: str, label: int) -> str:
    low = ch.lower()
    if low not in ALLOWED:
        return ch
    mod, tone = divmod(label, len(TONES))
    if mod not in ALLOWED[low]:
        mod = 0
    if low == "d":
        return ("Đ" if ch.isupper() else "đ") if mod == 4 else ch
    return unicodedata.normalize("NFC", ch + MODIFIERS[mod] + TONES[tone])


class DiacriticTagger:
    def __init__(self, model_dir: str | Path = ".", threads: int = 4, onnx_file: str = "tagger.int8.onnx"):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.index = {c: i + 2 for i, c in enumerate(cfg["vocab"])}  # 0 = padding, 1 = unknown
        self.max_len = cfg["model"]["max_len"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / onnx_file), opts, providers=["CPUExecutionProvider"])

    def restore(self, text: str) -> str:
        head, tail = text[: self.max_len], text[self.max_len :]  # longer input: the rest is copied
        if not head:
            return text
        ids = [self.index.get(c, 1) for c in head] + [0] * max(0, 2 - len(head))  # graph needs >= 2 positions
        logits = self.sess.run(["logits"], {"ids": np.array([ids], dtype=np.int64)})[0][0]
        out = []
        for ch, row in zip(head, logits):
            allowed = valid_labels(ch)
            out.append(apply_label(ch, allowed[int(np.argmax(row[allowed]))]))
        return "".join(out) + tail


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tagger = DiacriticTagger(Path(__file__).resolve().parent)
    for line in sys.argv[1:] or ["hom nay troi dep qua, chung ta di dao cong vien nhe"]:
        print(tagger.restore(line))
