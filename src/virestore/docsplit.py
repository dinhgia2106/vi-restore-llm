"""Leakage controls for the document-level split.

`virestore.data --split-by document` assigns whole documents to train / dev / test, then calls
`filter_leaks`, which removes every dev and test sentence that is a near-duplicate of a training
sentence: same text after lower-casing and masking digits, or a shared run of 8 words.

The `external` set comes from shards that were not used to build any split:

    python -m virestore.docsplit --corpus data/raw/wanjuan-external --train data/doc_split/train.jsonl \
        --out data/doc_split --per-domain 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from . import utf8_stdio

SHINGLE_WORDS = 8
_DIGITS = re.compile(r"\d")


def _words(sentence: str) -> list[str]:
    return _DIGITS.sub("0", sentence.lower()).split()


def _shingles(words: list[str]):
    for i in range(len(words) - SHINGLE_WORDS + 1):
        yield hash(" ".join(words[i : i + SHINGLE_WORDS]))


class TrainIndex:
    """Hashes of the masked training sentences and their 8-word shingles (valid in this process)."""

    def __init__(self, sentences):
        self.whole: set[int] = set()
        self.grams: set[int] = set()
        for s in sentences:
            w = _words(s)
            self.whole.add(hash(" ".join(w)))
            self.grams.update(_shingles(w))

    @classmethod
    def from_jsonl(cls, path: Path) -> "TrainIndex":
        def targets():
            with Path(path).open(encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)["target"]

        return cls(targets())

    def leaks(self, sentence: str) -> bool:
        w = _words(sentence)
        return hash(" ".join(w)) in self.whole or any(g in self.grams for g in _shingles(w))


def filter_leaks(out: Path, splits: tuple[str, ...] = ("dev", "test")) -> dict[str, dict[str, int]]:
    """Rewrite the given splits in `out` without near-duplicates of `out/train.jsonl`."""
    index = TrainIndex.from_jsonl(out / "train.jsonl")
    report = {}
    for name in splits:
        path, tmp = out / f"{name}.jsonl", out / f"{name}.jsonl.tmp"
        kept = removed = 0
        with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
            for line in src:
                if index.leaks(json.loads(line)["target"]):
                    removed += 1
                else:
                    dst.write(line)
                    kept += 1
        tmp.replace(path)
        report[name] = {"kept": kept, "removed": removed}
    return report


def _domain(relative: Path) -> str:
    """WanJuan layout: raw/<domain>/<source>/<shard>."""
    parts = relative.parts
    return parts[parts.index("raw") + 1] if "raw" in parts[:-1] else relative.stem


def build_external(
    corpus: Path, train: Path, out: Path, per_domain: int, per_doc: int, sample_rate: float, noisy: bool, seed: int,
    min_words: int = 3, max_words: int = 40,
) -> dict:
    """Sample sentences from unused shards: at most `per_doc` per document, none near a training sentence."""
    from .data import _iter_file, is_clean_sentence
    from .noise import NoiseConfig, make_input
    from .text import split_sentences

    index = TrainIndex.from_jsonl(train)
    rng = random.Random(seed)
    noise = NoiseConfig() if noisy else None
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    docs: Counter = Counter()
    removed = 0
    seen: set[bytes] = set()
    files = sorted(p for p in corpus.rglob("*.jsonl.gz"))
    with (out / "external.jsonl").open("w", encoding="utf-8") as dst:
        for f in files:
            domain = _domain(f.relative_to(corpus))
            for doc_id, doc in _iter_file(f, None, with_id=True):
                if counts[domain] >= per_domain:
                    break
                # a hash of the id picks documents spread over the shard, not just its first ones
                if int(hashlib.md5(doc_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF >= sample_rate:
                    continue
                taken = 0
                for paragraph in doc.splitlines():
                    for s in split_sentences(paragraph):
                        if taken >= per_doc or counts[domain] >= per_domain:
                            break
                        if not is_clean_sentence(s, min_words, max_words):
                            continue
                        key = hashlib.md5(s.encode("utf-8")).digest()[:8]
                        if key in seen:
                            continue
                        seen.add(key)
                        if index.leaks(s):
                            removed += 1
                            continue
                        row = {"input": make_input(s, rng, noise), "target": s, "domain": domain}
                        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[domain] += 1
                        taken += 1
                docs[domain] += taken > 0
    by_domain: dict[str, list[str]] = {}
    with (out / "external.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            by_domain.setdefault(json.loads(line)["domain"], []).append(line)
    for domain, lines in by_domain.items():
        (out / f"external_{domain}.jsonl").write_text("".join(lines), encoding="utf-8")
    stats = {
        "shards": [str(f.relative_to(corpus)) for f in files], "noisy": noisy, "per_doc": per_doc,
        "sample_rate": sample_rate, "sentences": dict(counts), "documents": dict(docs), "removed_as_near_train": removed,
    }
    (out / "external_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="folder of shards not used for train/dev/test")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-domain", type=int, default=2000)
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--sample-rate", type=float, default=0.1, help="share of documents considered")
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    stats = build_external(a.corpus, a.train, a.out, a.per_domain, a.per_doc, a.sample_rate, a.noisy, a.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
