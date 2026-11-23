"""Input-method wrapper around the character tagger: the rules VIMEK needs on live text.

The evaluation harness calls the tagger directly, so nothing here changes a reported number.

- Empty or whitespace-only input is returned as is, without calling the model.
- Text is split into sentences, and anything longer than MAX_CHUNK_CHARS at clause or word
  boundaries. A single token longer than that is copied unchanged.
- The model always sees fully stripped text, because marks in the input change its predictions.
  Any word the user already typed with marks is put back exactly as typed.
- A run of adjacent vowels never receives two tones.
"""

from __future__ import annotations

import re
import unicodedata

from .tagger import TONES, char_label
from .text import strip_diacritics

# Below the model's 384-character limit on purpose: it was trained on single sentences of at most
# 40 words and degrades on longer input.
MAX_CHUNK_CHARS = 192

_BREAKS = (re.compile(r"[.!?…]+[\"')\]]*\s+"), re.compile(r"[,;:]\s+"), re.compile(r"\s+"))
_TOKEN_AND_SPACE = re.compile(r"\S*\s*")
_WORDS = re.compile(r"(\s+)")


def split_chunks(text: str, max_len: int) -> list[tuple[str, bool]]:
    """Cut text into (piece, send_to_model); joining the pieces gives the text back.

    Every sentence is a separate piece even when the whole text fits the window: the model was
    trained on single sentences and degrades on multi-sentence input.
    """
    chunks: list[tuple[str, bool]] = []
    start = 0
    for m in _BREAKS[0].finditer(text):
        chunks += _split_long(text[start : m.end()], max_len)
        start = m.end()
    return chunks + _split_long(text[start:], max_len)


def _split_long(text: str, max_len: int) -> list[tuple[str, bool]]:
    chunks: list[tuple[str, bool]] = []
    start = 0
    while len(text) - start > max_len:
        window = text[start : start + max_len + 1]
        cut = 0
        for pattern in _BREAKS:
            ends = [m.end() if m.end() <= max_len else m.start() for m in pattern.finditer(window)]
            cut = max((e for e in ends if 0 < e <= max_len), default=0)
            if cut:
                break
        if cut:
            chunks.append((text[start : start + cut], True))
            start += cut
        else:
            end = _TOKEN_AND_SPACE.match(text, start).end()
            chunks.append((text[start:end], False))
            start = end
    if start < len(text):
        chunks.append((text[start:], True))
    return chunks


def has_stacked_tones(text: str) -> bool:
    """True if one run of adjacent vowels carries two or more tone marks (`thỏả`)."""
    tones = 0
    for ch in unicodedata.normalize("NFC", text):
        if strip_diacritics(ch).lower() in ("a", "e", "i", "o", "u", "y"):
            tones += char_label(ch) % len(TONES) > 0
            if tones > 1:
                return True
        else:
            tones = 0
    return False


def lock_typed_words(typed: str, restored: str) -> str:
    """Put back every whitespace-separated word the user typed with marks."""
    a, b = _WORDS.split(typed), _WORDS.split(restored)
    if len(a) != len(b):
        return restored
    return "".join(t if strip_diacritics(t) != t else r for t, r in zip(a, b))


class ImeRestorer:
    """Wraps a TaggerRestorer or OnnxTaggerRestorer."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.max_len = min(inner.max_len, MAX_CHUNK_CHARS)

    def _restore_chunks(self, plain: str) -> tuple[list[tuple[str, bool]], list[str]]:
        chunks = split_chunks(plain, self.max_len)
        todo = [piece for piece, send in chunks if send and piece.strip()]
        done = iter(self.inner.restore_one_tone(todo) if todo else [])
        return chunks, [next(done) if send and piece.strip() else piece for piece, send in chunks]

    def restore(self, inputs: list[str]) -> list[str]:
        out = []
        for raw in inputs:
            if not raw.strip():
                out.append(raw)
                continue
            typed = unicodedata.normalize("NFC", raw)
            _, parts = self._restore_chunks(strip_diacritics(typed))
            out.append(lock_typed_words(typed, "".join(parts)))
        return out

    def restore_nbest(self, inputs: list[str], k: int) -> list[list[str]]:
        """Candidates vary the last chunk, the part being typed; earlier chunks keep their best form."""
        out = []
        for raw in inputs:
            if not raw.strip():
                out.append([raw])
                continue
            typed = unicodedata.normalize("NFC", raw)
            chunks, parts = self._restore_chunks(strip_diacritics(typed))
            head, best = "".join(parts[:-1]), parts[-1]
            last, send = chunks[-1]
            alts = self.inner.restore_nbest([last], k * 3)[0] if send and last.strip() else []
            tails = [best] + [a for a in alts if not has_stacked_tones(a)]
            cands = [lock_typed_words(typed, head + t) for t in tails]
            out.append(list(dict.fromkeys(cands))[:k])
        return out
