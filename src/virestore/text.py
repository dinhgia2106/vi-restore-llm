"""Vietnamese text utilities: normalisation, diacritic stripping, sentence splitting."""

from __future__ import annotations

import re
import unicodedata

_COMBINING = re.compile(r"[̀-ͯ]")
_SPACES = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?…])\s+")

# Characters that are not decomposed by NFD but still carry a Vietnamese diacritic.
_SPECIAL = str.maketrans({"đ": "d", "Đ": "D"})


def normalize(text: str) -> str:
    """NFC-normalise and collapse whitespace."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text)).strip()


def strip_diacritics(text: str) -> str:
    """Remove every Vietnamese tone mark and vowel diacritic, keeping case and punctuation.

    >>> strip_diacritics("Tiếng Việt đẹp quá")
    'Tieng Viet dep qua'
    """
    decomposed = unicodedata.normalize("NFD", text.translate(_SPECIAL))
    return unicodedata.normalize("NFC", _COMBINING.sub("", decomposed))


def has_diacritics(text: str) -> bool:
    return strip_diacritics(text) != text


def split_sentences(text: str) -> list[str]:
    """Light-weight sentence splitter, good enough for building training pairs."""
    return [s.strip() for s in _SENT_END.split(normalize(text)) if s.strip()]


def words(text: str) -> list[str]:
    return text.split()
