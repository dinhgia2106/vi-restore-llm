"""Build a 'hard' test subset: sentences dense in ambiguous syllables.

A syllable is ambiguous when its no-diacritic form does not pin down the accented
form: even its most frequent accented form covers at most ``--max-top-share`` of
its occurrences in the training data (e.g. 'do' -> đó 44 %, độ 18 %, do 18 %, ...).

Almost every Vietnamese sentence contains a few such syllables (do, tu, thi, vi, ...),
so a sentence counts as hard when at least ``--min-ratio`` of its syllables are
ambiguous, not merely when it contains some.

    python -m virestore.hardset --train data/processed/train.jsonl --test data/processed/test.jsonl \
        --out data/processed/test_hard.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import utf8_stdio
from .data import load_jsonl
from .text import strip_diacritics

_PUNCT = ".,!?;:()\"'"


def ambiguous_syllables(train_rows: list[dict], max_top_share: float, min_total: int) -> set[str]:
    forms: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in train_rows:
        for w in r["target"].lower().split():
            w = w.strip(_PUNCT)
            if w:
                forms[strip_diacritics(w)][w] += 1
    amb = set()
    for plain, f in forms.items():
        total = sum(f.values())
        if total >= min_total and max(f.values()) / total <= max_top_share:
            amb.add(plain)
    return amb


def is_hard(text: str, amb: set[str], min_ratio: float, min_hits: int) -> bool:
    words = [w for w in (w.strip(_PUNCT) for w in text.lower().split()) if w]
    hits = sum(1 for w in words if w in amb)
    return hits >= min_hits and hits / max(len(words), 1) >= min_ratio


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-top-share", type=float, default=0.5, help="a syllable is ambiguous if its top accented form is at most this share")
    ap.add_argument("--min-total", type=int, default=50, help="ignore syllables seen fewer times than this in training")
    ap.add_argument("--min-ratio", type=float, default=0.3, help="share of a sentence's syllables that must be ambiguous")
    ap.add_argument("--min-hits", type=int, default=2, help="ambiguous syllables a sentence must contain")
    a = ap.parse_args()

    amb = ambiguous_syllables(load_jsonl(a.train), a.max_top_share, a.min_total)
    kept = total = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for r in load_jsonl(a.test):
            total += 1
            if is_hard(r["input"], amb, a.min_ratio, a.min_hits):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"ambiguous_syllables": len(amb), "hard_sentences": kept, "test_sentences": total, "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
