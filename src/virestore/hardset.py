"""Build a 'hard' test subset: sentences containing highly ambiguous syllables.

A syllable is ambiguous when its no-diacritic form maps to many accented forms in
the training data (e.g. 'ma' -> ma, mà, má, mả, mã, mạ).

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


def ambiguous_syllables(train_rows: list[dict], min_forms: int, min_count: int) -> set[str]:
    forms: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in train_rows:
        for w in r["target"].lower().split():
            w = w.strip(".,!?;:()\"'")
            if w:
                forms[strip_diacritics(w)][w] += 1
    return {
        plain
        for plain, f in forms.items()
        if sum(1 for c in f.values() if c >= min_count) >= min_forms
    }


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-forms", type=int, default=4, help="accented forms needed to call a syllable ambiguous")
    ap.add_argument("--min-count", type=int, default=5, help="ignore forms seen fewer times than this")
    ap.add_argument("--min-hits", type=int, default=2, help="ambiguous syllables a sentence must contain")
    a = ap.parse_args()

    amb = ambiguous_syllables(load_jsonl(a.train), a.min_forms, a.min_count)
    kept = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for r in load_jsonl(a.test):
            hits = sum(1 for w in r["input"].lower().split() if w.strip(".,!?;:()\"'") in amb)
            if hits >= a.min_hits:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"ambiguous_syllables": len(amb), "hard_sentences": kept, "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
