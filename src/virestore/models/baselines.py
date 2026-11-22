"""Cheap baselines every model has to beat."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..text import strip_diacritics


class Identity:
    """Returns the input unchanged — the floor."""

    name = "identity"

    def restore(self, inputs: list[str]) -> list[str]:
        return list(inputs)


class Lexicon:
    """Context-free lookup: each plain word -> its most frequent accented form in training data,
    with a bigram table to break ties between common readings (e.g. 'ma' -> 'mà' / 'ma' / 'mã')."""

    name = "lexicon"

    def __init__(self, unigram: dict[str, Counter], bigram: dict[tuple[str, str], Counter]):
        self.unigram = {k: c.most_common(1)[0][0] for k, c in unigram.items()}
        self.bigram = bigram

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: int | None = None) -> "Lexicon":
        unigram: dict[str, Counter] = defaultdict(Counter)
        bigram: dict[tuple[str, str], Counter] = defaultdict(Counter)
        with Path(path).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                words = json.loads(line)["target"].split()
                prev = "<s>"
                for w in words:
                    plain = strip_diacritics(w).lower()
                    unigram[plain][w.lower()] += 1
                    bigram[(prev, plain)][w.lower()] += 1
                    prev = w.lower()
        return cls(unigram, bigram)

    @staticmethod
    def _match_case(src: str, out: str) -> str:
        if src.isupper() and len(src) > 1:
            return out.upper()
        if src[:1].isupper():
            return out[:1].upper() + out[1:]
        return out

    def restore(self, inputs: list[str]) -> list[str]:
        results = []
        for sent in inputs:
            out, prev = [], "<s>"
            for w in sent.split():
                key = w.lower()
                cand = self.bigram.get((prev, key))
                best = cand.most_common(1)[0][0] if cand else self.unigram.get(key, key)
                out.append(self._match_case(w, best))
                prev = best
            results.append(" ".join(out))
        return results
