"""Evaluate a restorer the way an input method uses it: word by word, before the sentence is finished.

    python -m virestore.streaming --model tagger --tagger runs/ds-tagger \
        --test data/doc_split/test.jsonl --limit 2000 --batch-size 256 --tag ds-tagger-clean

The model is run on every word prefix of each test sentence. Commit policies are then simulated on
those outputs, so all of them are scored on the same predictions:

- `free`: every word on screen may still change;
- `wN`: only the last N+1 words may change, older ones are frozen (`w0` never revises a word).

Reported per policy: final word and sentence accuracy, accuracy when a word first appears, accuracy
averaged over all intermediate screens, and how much the text flickers (revisions per word, share of
words revised, revisions that fix or break a word, and how many words later a word settles).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from . import utf8_stdio

WINDOWS: tuple[int | None, ...] = (None, 0, 1, 2, 3, 5)
_WORD_END = re.compile(r"\S+")


def word_prefixes(text: str) -> list[str]:
    """Text up to the end of each word."""
    return [text[: m.end()] for m in _WORD_END.finditer(text)]


def policy_name(window: int | None) -> str:
    return "free" if window is None else f"w{window}"


def empty_counts() -> dict[str, int]:
    keys = ("sentences", "sentences_correct", "words", "final_correct", "first_correct", "screen_words", "screen_correct",
            "revisions", "words_revised", "helpful", "harmful", "settle_steps")
    return dict.fromkeys(keys, 0)


def simulate(steps: list[list[str]], target: list[str], window: int | None, c: dict[str, int]) -> None:
    """Add one sentence to the counts. `steps[k]` holds the model's k+1 words after the (k+1)-th word."""
    n = len(target)
    shown: list[str] = []
    last_change = [0] * n
    changes = [0] * n
    for k in range(n):
        for j in range(k):
            if window is None or j >= k - window:
                new = steps[k][j]
                if new != shown[j]:
                    changes[j] += 1
                    last_change[j] = k - j
                    c["revisions"] += 1
                    c["helpful"] += new == target[j]
                    c["harmful"] += shown[j] == target[j]
                    shown[j] = new
        shown.append(steps[k][k])
        c["first_correct"] += shown[k] == target[k]
        c["screen_words"] += k + 1
        c["screen_correct"] += sum(a == b for a, b in zip(shown, target))
    c["sentences"] += 1
    c["sentences_correct"] += shown == target
    c["words"] += n
    c["final_correct"] += sum(a == b for a, b in zip(shown, target))
    c["words_revised"] += sum(x > 0 for x in changes)
    c["settle_steps"] += sum(last_change)


def rates(c: dict[str, int]) -> dict[str, float]:
    w = max(c["words"], 1)
    return {
        "sent_acc": round(c["sentences_correct"] / max(c["sentences"], 1), 4),
        "final_word_acc": round(c["final_correct"] / w, 4),
        "first_shown_word_acc": round(c["first_correct"] / w, 4),
        "screen_word_acc": round(c["screen_correct"] / max(c["screen_words"], 1), 4),
        "revisions_per_100_words": round(100 * c["revisions"] / w, 2),
        "words_revised": round(c["words_revised"] / w, 4),
        "helpful_revisions": c["helpful"],
        "harmful_revisions": c["harmful"],
        "mean_settle_words": round(c["settle_steps"] / w, 4),
    }


def evaluate(restorer, rows: list[dict], batch_size: int) -> tuple[dict, int]:
    """Run all prefixes (longest first, for tight batches) and score every policy."""
    jobs = [(len(p), i, k, p) for i, r in enumerate(rows) for k, p in enumerate(word_prefixes(r["input"]))]
    jobs.sort(key=lambda x: -x[0])
    outputs: list[list[list[str] | None]] = [[None] * len(r["input"].split()) for r in rows]
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b : b + batch_size]
        for (_, i, k, _), out in zip(chunk, restorer.restore([j[3] for j in chunk])):
            outputs[i][k] = out.split()
    counts = {policy_name(w): empty_counts() for w in WINDOWS}
    skipped = 0
    for r, steps in zip(rows, outputs):
        target = r["target"].split()
        if len(target) != len(steps) or any(s is None or len(s) != k + 1 for k, s in enumerate(steps)):
            skipped += 1  # a model that changes the word count cannot be aligned to the screen
            continue
        for w in WINDOWS:
            simulate(steps, target, w, counts[policy_name(w)])
    return {name: {**rates(c), "n_sentences": c["sentences"], "n_words": c["words"]} for name, c in counts.items()}, skipped


def keystroke_latency(restorer, rows: list[dict], n: int) -> dict:
    """Single-request time of one call per word, over the prefixes of the first `n` sentences."""
    times = []
    for r in rows[:n]:
        for p in word_prefixes(r["input"]):
            t0 = time.perf_counter()
            restorer.restore([p])
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    if not times:
        return {}
    return {
        "calls": len(times),
        "p50_ms": round(times[int(0.5 * (len(times) - 1))], 2),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 2),
        "max_ms": round(times[-1], 2),
    }


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["lexicon", "tcn", "tagger", "tagger-onnx"])
    ap.add_argument("--tagger")
    ap.add_argument("--onnx-file", help="ONNX file inside --tagger for tagger-onnx (default: tagger.onnx)")
    ap.add_argument("--train", help="training JSONL (lexicon)")
    ap.add_argument("--tcn-repo")
    ap.add_argument("--tcn-weights")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--latency-n", type=int, default=0, help="sentences whose prefixes are timed one call at a time")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()

    from . import models
    from .data import load_jsonl
    from .eval import machine_info

    restorer = models.load(a.model, model=a.tagger, train=a.train, tcn_repo=a.tcn_repo, tcn_weights=a.tcn_weights,
                           threads=a.threads, batch_size=a.batch_size, onnx_file=a.onnx_file)
    rows = load_jsonl(a.test, a.limit)
    policies, skipped = evaluate(restorer, rows, a.batch_size)
    out = {
        "tag": a.tag, "model": restorer.name, "test_file": str(a.test), "n": len(rows), "skipped_unaligned": skipped,
        "policies": policies, "keystroke_latency": keystroke_latency(restorer, rows, a.latency_n) if a.latency_n else None,
        "machine": machine_info(restorer),
    }
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"{a.tag}.stream.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
