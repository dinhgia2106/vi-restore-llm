"""Build (input, target) training pairs from raw Vietnamese text.

Usage:
    python -m virestore.data --corpus data/raw --out data/processed --noisy
    python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed --limit 2000000

The corpus can be a .txt file, a .jsonl(.gz) file, or a folder containing either.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from . import utf8_stdio
from .noise import NoiseConfig, make_input
from .text import has_diacritics, split_sentences

SPLITS = (("train", 0.98), ("dev", 0.01), ("test", 0.01))


def _split_of(sentence: str) -> str:
    """Deterministic split by content hash, so a sentence never leaks across splits."""
    h = int(hashlib.md5(sentence.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if h < acc:
            return name
    return SPLITS[-1][0]


TEXT_SUFFIXES = (".txt", ".jsonl", ".jsonl.gz", ".json.gz")
TEXT_FIELDS = ("content", "text")  # WanJuan-Vietnamese uses JSON Lines; field is auto-detected
_JUNK = re.compile(r"https?://|www\.|@|\{|\}|<[a-z/]|\|")


def _open(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open(encoding="utf-8", errors="ignore")


def _iter_file(f: Path, text_field: str | None):
    is_json = ".json" in f.name
    with _open(f) as fh:
        for line in fh:
            if not is_json:
                yield line
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            field = text_field or next((k for k in TEXT_FIELDS if k in obj), None)
            if field and isinstance(obj.get(field), str):
                yield obj[field]


def iter_documents(corpus: Path, text_field: str | None = None):
    """Yield raw text from .txt files (one document per line) or JSON Lines files.

    Documents from several files are interleaved round-robin, so that ``--limit``
    draws from every shard (and every WanJuan domain) instead of only the first one.
    """
    files = [corpus] if corpus.is_file() else sorted(p for p in corpus.rglob("*") if p.name.endswith(TEXT_SUFFIXES))
    streams = [_iter_file(f, text_field) for f in files]
    while streams:
        alive = []
        for it in streams:
            doc = next(it, None)
            if doc is not None:
                yield doc
                alive.append(it)
        streams = alive


def is_clean_sentence(s: str, min_words: int, max_words: int) -> bool:
    n = len(s.split())
    if not (min_words <= n <= max_words) or not has_diacritics(s) or _JUNK.search(s):
        return False
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    return letters / max(len(s), 1) >= 0.7 and digits / max(len(s), 1) <= 0.1


def iter_sentences(corpus: Path, min_words: int, max_words: int, text_field: str | None = None):
    for doc in iter_documents(corpus, text_field):
        for paragraph in doc.splitlines():
            for s in split_sentences(paragraph):
                if is_clean_sentence(s, min_words, max_words):
                    yield s


def build(
    corpus: Path, out: Path, noisy: bool, seed: int, min_words: int, max_words: int, limit: int | None, text_field: str | None = None
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    noise = NoiseConfig() if noisy else None
    writers = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name, _ in SPLITS}
    seen: set[bytes] = set()  # 8-byte digests keep memory small on millions of sentences
    counts = {name: 0 for name, _ in SPLITS}
    try:
        for s in iter_sentences(corpus, min_words, max_words, text_field):
            key = hashlib.md5(s.encode("utf-8")).digest()[:8]
            if key in seen:
                continue
            seen.add(key)
            split = _split_of(s)
            row = {"input": make_input(s, rng, noise), "target": s}
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            if limit and len(seen) >= limit:
                break
    finally:
        for w in writers.values():
            w.close()
    stats = {"corpus": str(corpus), "noisy": noisy, "seed": seed, "min_words": min_words, "max_words": max_words, "counts": counts}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="a .txt file or a folder of .txt files")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--noisy", action="store_true", help="add Telex/typo noise on top of diacritic stripping")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="stop after N unique sentences")
    ap.add_argument("--text-field", default=None, help="JSON field holding the text (default: auto-detect content/text)")
    args = ap.parse_args()
    stats = build(args.corpus, args.out, args.noisy, args.seed, args.min_words, args.max_words, args.limit, args.text_field)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
