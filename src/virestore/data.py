"""Build (input, target) training pairs from raw Vietnamese text.

Usage:
    python -m virestore.data --corpus data/raw --out data/processed --noisy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def iter_sentences(corpus: Path, min_words: int, max_words: int):
    files = sorted(corpus.rglob("*.txt")) if corpus.is_dir() else [corpus]
    for f in files:
        with f.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for s in split_sentences(line):
                    n = len(s.split())
                    if min_words <= n <= max_words and has_diacritics(s):
                        yield s


def build(corpus: Path, out: Path, noisy: bool, seed: int, min_words: int, max_words: int, limit: int | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    noise = NoiseConfig() if noisy else None
    writers = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name, _ in SPLITS}
    seen: set[str] = set()
    counts = {name: 0 for name, _ in SPLITS}
    try:
        for s in iter_sentences(corpus, min_words, max_words):
            key = hashlib.md5(s.encode("utf-8")).hexdigest()
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
    stats = {"corpus": str(corpus), "noisy": noisy, "seed": seed, "counts": counts}
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
    args = ap.parse_args()
    stats = build(args.corpus, args.out, args.noisy, args.seed, args.min_words, args.max_words, args.limit)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
