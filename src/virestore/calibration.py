"""When can an input method apply the tagger's output without asking?

    python -m virestore.calibration --tagger runs/ds-tagger --dev data/doc_split/dev.jsonl \
        --test data/doc_split/test.jsonl --tag ds-tagger-clean

Temperature and decision thresholds are fitted on dev only. A word's confidence is the product of
the top-label probabilities of its letters; a sentence's is the product over all its letters. On
test the script reports calibration (ECE, Brier), ranking (AUROC), and, for each target precision,
how many words or sentences could be auto-applied at the dev threshold and how precise they are.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import utf8_stdio

TARGETS = (0.90, 0.95, 0.98, 0.99)
TEMPERATURES = [round(0.5 + 0.05 * i, 2) for i in range(71)]  # 0.50 ... 4.00


def ece(conf, correct, bins: int = 15) -> float:
    import numpy as np

    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def auroc(score, positive) -> float:
    """Probability that a correct item outranks a wrong one (ties count half)."""
    import numpy as np

    score, positive = np.asarray(score, dtype=float), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # average rank of tied scores
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_for(conf, correct, target: float) -> float | None:
    """Lowest threshold whose accepted items reach the target precision on this (dev) data."""
    import numpy as np

    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    order = np.argsort(-conf, kind="stable")
    precision = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    # a threshold can only fall between distinct confidence values
    distinct = np.r_[conf[order][1:] != conf[order][:-1], True]
    ok = np.flatnonzero((precision >= target) & distinct)
    return float(conf[order][ok[-1]]) if len(ok) else None


def collect(restorer, rows: list[dict], temperature: float, batch_size: int) -> dict:
    """Confidence and correctness of each sentence and word."""
    import numpy as np
    import torch

    from .tagger import TAGGABLE, decode

    sent_conf, sent_ok, word_conf, word_ok = [], [], [], []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input"]))
    for b in range(0, len(order), batch_size):
        chunk = [rows[i] for i in order[b : b + batch_size]]
        lp, n = restorer._log_probs([r["input"] for r in chunk])
        scaled = torch.log_softmax(lp / temperature, dim=-1)
        top, labels = scaled.max(-1)
        for k, r in enumerate(chunk):
            src = r["input"][:n]
            out = decode(src, labels[k, : len(src)].tolist()) + r["input"][n:]
            logs = top[k, : len(src)].tolist()
            total, word_log, start_ok = 0.0, 0.0, 0
            out_words, ref_words = out.split(), r["target"].split()
            aligned = len(out_words) == len(ref_words)
            idx = 0
            for j, c in enumerate(src + " "):
                if c == " ":
                    if j > start_ok:
                        if aligned and idx < len(ref_words):
                            word_conf.append(np.exp(word_log))
                            word_ok.append(out_words[idx] == ref_words[idx])
                        idx += 1
                    word_log, start_ok = 0.0, j + 1
                elif c in TAGGABLE:
                    word_log += logs[j]
                    total += logs[j]
            sent_conf.append(np.exp(total))
            sent_ok.append(out == r["target"])
    return {"sentence": (np.array(sent_conf), np.array(sent_ok)), "word": (np.array(word_conf), np.array(word_ok))}


def fit_temperature(restorer, rows: list[dict], batch_size: int, max_chars: int = 400_000) -> tuple[float, float]:
    """Grid search for the temperature with the lowest NLL on the letters of dev sentences."""
    import torch

    from .tagger import TAGGABLE, labels_for

    logits, gold = [], []
    seen = 0
    for b in range(0, len(rows), batch_size):
        chunk = rows[b : b + batch_size]
        lp, n = restorer._log_probs([r["input"] for r in chunk])
        for k, r in enumerate(chunk):
            target = labels_for(r["target"])
            src = r["input"][:n]
            if len(target) != len(r["input"]):
                continue
            keep = [j for j, c in enumerate(src) if c in TAGGABLE]
            logits.append(lp[k, keep])
            gold += [target[j] for j in keep]
            seen += len(keep)
        if seen >= max_chars:
            break
    x, y = torch.cat(logits), torch.tensor(gold)
    valid = torch.isfinite(x[torch.arange(len(y)), y])
    x, y = x[valid], y[valid]

    def nll(t: float) -> float:
        lp_t = torch.log_softmax(x / t, dim=-1)
        return float(-lp_t[torch.arange(len(y)), y].mean())

    best = min(TEMPERATURES, key=nll)
    return best, nll(best)


def report(dev: tuple, test: tuple) -> dict:
    import numpy as np

    conf, ok = test
    out = {
        "n": int(len(ok)), "accuracy": round(float(ok.mean()), 4), "ece": round(ece(conf, ok), 4),
        "brier": round(float(np.mean((conf - ok.astype(float)) ** 2)), 4), "auroc": round(auroc(conf, ok), 4), "auto_apply": {},
    }
    for target in TARGETS:
        th = threshold_for(dev[0], dev[1], target)
        if th is None:
            out["auto_apply"][str(target)] = None
            continue
        m = conf >= th
        out["auto_apply"][str(target)] = {
            "dev_threshold": round(th, 6), "test_coverage": round(float(m.mean()), 4),
            "test_precision": round(float(ok[m].mean()), 4) if m.any() else None,
        }
    return out


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tagger", required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()

    from .data import load_jsonl
    from .models.tagger import TaggerRestorer

    restorer = TaggerRestorer(a.tagger, batch_size=a.batch_size)
    dev, test = load_jsonl(a.dev), load_jsonl(a.test)
    temperature, dev_nll = fit_temperature(restorer, dev, a.batch_size)
    out = {"tag": a.tag, "tagger": str(a.tagger), "dev_file": str(a.dev), "test_file": str(a.test),
           "temperature": temperature, "dev_nll": round(dev_nll, 5)}
    for name, t in (("uncalibrated", 1.0), ("temperature_scaled", temperature)):
        d, te = collect(restorer, dev, t, a.batch_size), collect(restorer, test, t, a.batch_size)
        out[name] = {level: report(d[level], te[level]) for level in ("word", "sentence")}
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"{a.tag}.calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
