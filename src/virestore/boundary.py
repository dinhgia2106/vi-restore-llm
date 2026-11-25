"""Word boundaries for text typed without spaces (and without marks).

    python -m virestore.boundary --config configs/boundary_ds.yaml                       # train
    python -m virestore.boundary --eval runs/ds-boundary --test data/doc_split/test.jsonl  # boundary F1

The model sees a sentence with marks and whitespace removed (`homnaytroidepqua`) and labels each
character with 1 if a space follows it. It reuses the tagger's character Transformer with a
two-class head. The diacritic tagger then marks the spaced text; `virestore.eval --model
segment-tagger` scores the two together against the ordinary references.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import time
from pathlib import Path

from . import utf8_stdio
from .text import strip_diacritics

IGNORE = -100


def boundary_pair(text: str) -> tuple[str, list[int]]:
    """`hôm nay trời` -> ("homnaytroi", [0,0,1,0,0,1,0,0,0,0]): 1 where a space follows.

    The last character is labelled 0 here; training ignores it, since whether a space follows the
    last typed character is unknown.
    """
    words = strip_diacritics(text).split()
    src = "".join(words)
    labels = [0] * len(src)
    end = 0
    for w in words[:-1]:
        end += len(w)
        labels[end - 1] = 1
    return src, labels


def join(src: str, labels: list[int]) -> str:
    """Insert a space after every character labelled 1."""
    return "".join(c + (" " if l == 1 and i + 1 < len(src) else "") for i, (c, l) in enumerate(zip(src, labels)))


def load_rows(path: Path, limit: int | None = None) -> list[tuple[str, list[int]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            src, labels = boundary_pair(r.get("input") or r["target"])  # marks are stripped either way
            if src:
                rows.append((src, labels))
    return rows


def _batches(rows, vocab, batch_size, max_len, shuffle, rng, prefix_prob: float = 0.0):
    import torch

    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = []
        for j in order[i : i + batch_size]:
            src, labels = rows[j]
            if prefix_prob and len(src) > 1 and rng.random() < prefix_prob:
                cut = rng.randint(1, len(src) - 1)  # the user is still typing, possibly mid-word
                src, labels = src[:cut], labels[:cut]
            chunk.append((src, labels))
        n = min(max(len(s) for s, _ in chunk), max_len)
        ids = torch.zeros((len(chunk), n), dtype=torch.long)
        lab = torch.full((len(chunk), n), IGNORE, dtype=torch.long)
        for k, (src, labels) in enumerate(chunk):
            src, labels = src[:n], labels[:n]
            ids[k, : len(src)] = torch.tensor(vocab.encode(src))
            lab[k, : len(src) - 1] = torch.tensor(labels[: len(src) - 1], dtype=torch.long)
        yield ids, ids != 0, lab


def boundary_scores(pred_rows: list[list[int]], gold_rows: list[list[int]]) -> dict:
    """Precision / recall / F1 of predicted spaces, and the share of sentences segmented exactly."""
    tp = fp = fn = exact = 0
    for p, g in zip(pred_rows, gold_rows):
        p, g = p[: len(g) - 1], g[: len(g) - 1]
        tp += sum(1 for a, b in zip(p, g) if a == 1 and b == 1)
        fp += sum(1 for a, b in zip(p, g) if a == 1 and b == 0)
        fn += sum(1 for a, b in zip(p, g) if a == 0 and b == 1)
        exact += p == g
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
        "sentence_exact": round(exact / max(len(gold_rows), 1), 6),
        "n": len(gold_rows),
    }


class BoundaryModel:
    """A trained boundary model: per-character probability that a space follows."""

    def __init__(self, path: str, threads: int | None = None, batch_size: int = 64):
        import torch

        from .tagger import Vocab, build_model

        if threads:
            torch.set_num_threads(threads)
        self.device = "cuda" if torch.cuda.is_available() and not threads else "cpu"
        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        m = self.cfg["model"]
        self.vocab = Vocab(self.cfg["vocab"])
        self.model = build_model(len(self.vocab), m["d_model"], m["layers"], m["heads"], m["ff"], m["max_len"], 0.0, n_labels=2)
        self.model.load_state_dict(torch.load(p / "model.pt", map_location=self.device))
        self.model.to(self.device).eval()
        self.max_len, self.batch_size = m["max_len"], batch_size
        self.name = f"boundary:{p.name}"

    def probs(self, texts: list[str]) -> list[list[float]]:
        import torch

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[: self.max_len] for t in texts[i : i + self.batch_size]]
            n = max((len(t) for t in batch), default=0)
            if n == 0:
                out += [[] for _ in batch]
                continue
            ids = torch.zeros((len(batch), n), dtype=torch.long)
            for k, t in enumerate(batch):
                if t:
                    ids[k, : len(t)] = torch.tensor(self.vocab.encode(t))
            ids = ids.to(self.device)
            with torch.inference_mode():
                p = torch.softmax(self.model(ids, ids != 0).float(), -1)[..., 1].cpu()
            out += [p[k, : len(t)].tolist() for k, t in enumerate(batch)]
        return out

    def segment(self, texts: list[str]) -> list[str]:
        """Best segmentation of each text; characters beyond max_len are appended unsplit."""
        return [join(t, [int(x > 0.5) for x in p]) + t[len(p) :] for t, p in zip(texts, self.probs(texts))]

    def segment_nbest(self, text: str, k: int) -> list[str]:
        """Top-k segmentations by beam search over independent per-character decisions."""
        p = self.probs([text])[0]
        if not p:
            return [text]
        beam: list[tuple[float, list[int]]] = [(0.0, [])]
        for x in p[:-1]:
            x = min(max(x, 1e-9), 1 - 1e-9)
            options = []
            for s, lab in beam:
                options.append((s + math.log(1 - x), lab + [0]))
                options.append((s + math.log(x), lab + [1]))
            beam = heapq.nlargest(max(4 * k, 16), options, key=lambda o: o[0])
        segs = [join(text[: len(p)], lab + [0]) + text[len(p) :] for _, lab in beam]
        return list(dict.fromkeys(segs))[:k]


class SegmentTagger:
    """Restorer for unspaced input: boundaries first, then the diacritic tagger.

    For evaluation on the usual test files, the spaces in `input` are removed first, so the
    output is scored against the ordinary spaced references.
    """

    def __init__(self, boundary_dir: str, tagger_dir: str, threads: int | None = None, batch_size: int = 64,
                 onnx_file: str | None = None):
        from .models.tagger import OnnxTaggerRestorer, TaggerRestorer

        self.boundary = BoundaryModel(boundary_dir, threads=threads, batch_size=batch_size)
        if onnx_file:
            self.tagger = OnnxTaggerRestorer(tagger_dir, threads=threads, batch_size=batch_size, onnx_file=onnx_file)
        else:
            self.tagger = TaggerRestorer(tagger_dir, threads=threads, batch_size=batch_size)
        self.name = f"segment-tagger:{Path(boundary_dir).name}+{Path(tagger_dir).name}"

    def restore(self, inputs: list[str]) -> list[str]:
        spaced = self.boundary.segment(["".join(t.split()) for t in inputs])
        return self.tagger.restore(spaced)

    def restore_nbest(self, inputs: list[str], k: int) -> list[list[str]]:
        out = []
        for t in inputs:
            segs = self.boundary.segment_nbest("".join(t.split()), k)
            cands = self.tagger.restore_nbest([segs[0]], k)[0] + self.tagger.restore(segs[1:])
            out.append(list(dict.fromkeys(cands))[:k])
        return out


def train(cfg: dict) -> None:
    import torch

    from .tagger import Vocab, build_model, save

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    rng = random.Random(cfg.get("seed", 13))
    torch.manual_seed(cfg.get("seed", 13))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(Path(d["train"]), d.get("train_limit"))
    dev = load_rows(Path(d["dev"]), d.get("dev_limit", 5000))
    counts: dict[str, int] = {}
    for src, _ in rows[:500_000]:
        for c in src:
            counts[c] = counts.get(c, 0) + 1
    vocab = Vocab(sorted(c for c, n in counts.items() if n >= d.get("min_char_count", 20)))
    print(f"train {len(rows)} dev {len(dev)} vocab {len(vocab)} device {device}", flush=True)
    model = build_model(len(vocab), m["d_model"], m["layers"], m["heads"], m["ff"], m["max_len"], m["dropout"], n_labels=2).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    steps = t["epochs"] * math.ceil(len(rows) / t["batch_size"])
    warm = int(steps * t.get("warmup_ratio", 0.02))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    out = Path(cfg["output_dir"])
    best, step, t0 = -1.0, 0, time.time()

    def evaluate() -> dict:
        model.eval()
        preds, golds = [], []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for ids, mask, lab in _batches(dev, vocab, 512, m["max_len"], False, rng):
                pred = model(ids.to(device), mask.to(device)).argmax(-1).cpu()
                for k in range(ids.shape[0]):
                    n = int(mask[k].sum())
                    preds.append(pred[k, :n].tolist())
                    golds.append([max(x, 0) for x in lab[k, :n].tolist()])
        model.train()
        return boundary_scores(preds, golds)

    model.train()
    for epoch in range(t["epochs"]):
        for ids, mask, lab in _batches(rows, vocab, t["batch_size"], m["max_len"], True, rng, d.get("prefix_prob", 0.0)):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(ids, mask)
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 2), lab.reshape(-1), ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % t["log_steps"] == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % t["eval_steps"] == 0 or step == steps:
                s = evaluate()
                print(f"  dev boundary {s}", flush=True)
                if s["f1"] > best:
                    best = s["f1"]
                    save(model, vocab, cfg, out)
    print(f"best dev boundary F1 {best:.6f}; saved to {out}", flush=True)


def evaluate_dir(model_dir: str, test: Path, limit: int | None, tag: str, results: Path) -> dict:
    rows = load_rows(test, limit)
    bm = BoundaryModel(model_dir)
    t0 = time.perf_counter()
    preds = [[int(x > 0.5) for x in p] for p in bm.probs([s for s, _ in rows])]
    scores = boundary_scores(preds, [g for _, g in rows])
    rec = {"tag": tag, "model": bm.name, "test": str(test), "device": bm.device, "boundary": scores,
           "seconds": round(time.perf_counter() - t0, 2)}
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{tag}.boundary.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="train with this YAML config")
    ap.add_argument("--eval", help="evaluate this trained model directory")
    ap.add_argument("--test", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    if a.config:
        import yaml

        train(yaml.safe_load(a.config.read_text(encoding="utf-8")))
    elif a.eval and a.test:
        print(json.dumps(evaluate_dir(a.eval, a.test, a.limit, a.tag or f"boundary-{Path(a.eval).name}", a.results), indent=1))
    else:
        ap.error("give --config to train, or --eval and --test to evaluate")


if __name__ == "__main__":
    main()
