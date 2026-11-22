"""Evaluation metrics for restoration outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .text import normalize


def levenshtein(a: list | str, b: list | str) -> int:
    """Edit distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@dataclass
class Scores:
    n: int
    cer: float  # character error rate
    wer: float  # word error rate
    word_acc: float  # position-wise word accuracy (same word count only)
    sent_acc: float  # exact sentence match

    def as_dict(self) -> dict:
        return {k: round(v, 5) if isinstance(v, float) else v for k, v in asdict(self).items()}


def score(predictions: list[str], references: list[str]) -> Scores:
    assert len(predictions) == len(references), "prediction / reference count mismatch"
    char_err = char_tot = word_err = word_tot = 0
    aligned_ok = aligned_tot = exact = 0
    for pred, ref in zip(predictions, references):
        pred, ref = normalize(pred), normalize(ref)
        char_err += levenshtein(pred, ref)
        char_tot += max(len(ref), 1)
        pw, rw = pred.split(), ref.split()
        word_err += levenshtein(pw, rw)
        word_tot += max(len(rw), 1)
        if len(pw) == len(rw):
            aligned_ok += sum(p == r for p, r in zip(pw, rw))
            aligned_tot += len(rw)
        exact += pred == ref
    n = len(references)
    return Scores(
        n=n,
        cer=char_err / max(char_tot, 1),
        wer=word_err / max(word_tot, 1),
        word_acc=aligned_ok / max(aligned_tot, 1),
        sent_acc=exact / max(n, 1),
    )
