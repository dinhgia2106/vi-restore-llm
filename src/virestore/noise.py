"""Synthetic typing noise for no-diacritic Vietnamese input.

The clean task is pure diacritic restoration (input = target with marks stripped).
The noisy task adds the mistakes people actually make when typing fast with an IME:
leftover Telex keystrokes, dropped or doubled letters, swapped neighbours.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

from .text import strip_diacritics

# Telex keystrokes that produce each mark; used to simulate "unconverted" Telex.
_TONE_KEYS = {"́": "s", "̀": "f", "̉": "r", "̃": "x", "̣": "j"}
_VOWEL_KEYS = {"â": "aa", "ê": "ee", "ô": "oo", "ă": "aw", "ơ": "ow", "ư": "uw", "đ": "dd"}


@dataclass(frozen=True)
class NoiseConfig:
    p_telex: float = 0.03  # word keeps its raw Telex keystrokes, e.g. "tieengs"
    p_drop: float = 0.01  # drop one letter
    p_double: float = 0.01  # repeat one letter
    p_swap: float = 0.01  # swap two neighbouring letters


def telex_spelling(word: str) -> str:
    """Spell a word the way it is typed in Telex: 'tiếng' -> 'tieengs'."""
    out, tone = [], ""
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch in _TONE_KEYS:
            tone = _TONE_KEYS[ch]
            continue
        out.append(ch)
    base = unicodedata.normalize("NFC", "".join(out))
    typed = "".join(_VOWEL_KEYS.get(c, c) for c in base)
    # NFD keeps the breve/horn/circumflex as combining marks; remove leftovers.
    return strip_diacritics(typed) + tone


def _letters(word: str) -> list[int]:
    return [i for i, c in enumerate(word) if c.isalpha()]


def corrupt_word(word: str, rng: random.Random, cfg: NoiseConfig) -> str:
    plain = strip_diacritics(word)
    r = rng.random()
    if r < cfg.p_telex and plain.isalpha() and plain != word:
        return telex_spelling(word)
    r -= cfg.p_telex
    idx = _letters(plain)
    if len(idx) < 2:
        return plain
    if r < cfg.p_drop:
        i = rng.choice(idx)
        return plain[:i] + plain[i + 1 :]
    r -= cfg.p_drop
    if r < cfg.p_double:
        i = rng.choice(idx)
        return plain[: i + 1] + plain[i] + plain[i + 1 :]
    r -= cfg.p_double
    if r < cfg.p_swap:
        i = rng.choice(idx[:-1])
        return plain[:i] + plain[i + 1] + plain[i] + plain[i + 2 :]
    return plain


def make_input(target: str, rng: random.Random, cfg: NoiseConfig | None = None) -> str:
    """Build the model input for a clean target sentence.

    With ``cfg=None`` this is exact diacritic stripping (the clean task).
    """
    if cfg is None:
        return strip_diacritics(target)
    return " ".join(corrupt_word(w, rng, cfg) for w in target.split())
