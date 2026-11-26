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


def word_span(src: str, labels: list[int], rng, max_words: int = 8) -> tuple[str, list[int]]:
    """A random run of 1..max_words consecutive words, as an input method sees a chunk typed in the
    middle of a sentence."""
    ends = [i for i, l in enumerate(labels[:-1]) if l == 1] + [len(src) - 1]
    starts = [0] + [e + 1 for e in ends[:-1]]
    first = rng.randrange(len(starts))
    last = min(first + rng.randint(1, max_words), len(starts)) - 1
    a, b = starts[first], ends[last] + 1
    return src[a:b], labels[a:b]


def _batches(rows, vocab, batch_size, max_len, shuffle, rng, prefix_prob: float = 0.0, span_prob: float = 0.0):
    import torch

    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = []
        for j in order[i : i + batch_size]:
            src, labels = rows[j]
            if span_prob and rng.random() < span_prob:
                src, labels = word_span(src, labels, rng)
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


MAX_TOKEN = 24  # longest letter token the lexicon decoder considers


def build_lexicon(train: Path, min_count: int = 3, limit: int | None = None) -> dict[str, int]:
    """Lower-case letter tokens of the training inputs (marks removed) seen at least min_count times."""
    counts: dict[str, int] = {}
    with train.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            for w in strip_diacritics(r.get("input") or r["target"]).lower().split():
                if w.isalpha() and len(w) <= MAX_TOKEN:
                    counts[w] = counts.get(w, 0) + 1
    return {w: n for w, n in counts.items() if n >= min_count}


def _letter_runs(src: str) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(src):
        if src[i].isalpha():
            j = i
            while j < len(src) and src[j].isalpha():
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def lexicon_decode(src: str, probs: list[float], lexicon, unknown_logp: float, k: int = 1) -> list[tuple[float, list[int]]]:
    """Top-k label sequences where every letter token is a known token or pays `unknown_logp`.

    Inside each run of letters the cut points are chosen by dynamic programming over the model's
    log-probabilities; boundaries next to digits or punctuation keep the model's own decision.
    """
    n = len(probs)
    if n == 0:
        return [(0.0, [])]
    p = [min(max(x, 1e-9), 1 - 1e-9) for x in probs]
    base = [int(x > 0.5) for x in p]
    base[-1] = 0
    keep = [math.log(1 - x) for x in p]
    cut = [math.log(x) for x in p]
    pre = [0.0]
    for v in keep:
        pre.append(pre[-1] + v)
    beam: list[tuple[float, list[int]]] = [(0.0, base)]
    for a, b in _letter_runs(src[:n]):
        # best[j]: top-k (score, cut positions) for src[a:j] cut after position j-1 (if j < b)
        best: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(b + 1)]
        best[a] = [(0.0, ())]
        for j in range(a + 1, b + 1):
            options = []
            for s in range(max(a, j - MAX_TOKEN), j):
                if not best[s]:
                    continue
                seg = src[s:j].lower()
                score = pre[j - 1] - pre[s] + (cut[j - 1] if j < b else 0.0) + (0.0 if seg in lexicon else unknown_logp)
                for sc, cuts in best[s]:
                    options.append((sc + score, cuts + ((j - 1,) if j < b else ())))
            best[j] = heapq.nlargest(k, options, key=lambda o: o[0])
        if not best[b]:
            continue
        merged = []
        for sc, labels in beam:
            for run_sc, cuts in best[b]:
                lab = list(labels)
                for i in range(a, b - 1):
                    lab[i] = 0
                for i in cuts:
                    lab[i] = 1
                merged.append((sc + run_sc, lab))
        beam = heapq.nlargest(k, merged, key=lambda o: o[0])
    return beam


class BoundaryModel:
    """A trained boundary model: per-character probability that a space follows.

    With use_lexicon and a `lexicon.json` next to the model (from `--build-lexicon`), each run of letters
    is cut so that its tokens are known ones where the model allows it (`lexicon_decode`).
    """

    def __init__(self, path: str, threads: int | None = None, batch_size: int = 64, use_lexicon: bool = False,
                 unknown_logp: float = -8.0):
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
        lex = p / "lexicon.json"
        self.lexicon = json.loads(lex.read_text(encoding="utf-8")) if use_lexicon and lex.exists() else None  # token -> count
        self.unknown_logp = unknown_logp
        self.name = f"boundary:{p.name}" + ("+lexicon" if self.lexicon else "")

    def labels(self, text: str, probs: list[float], k: int = 1) -> list[list[int]]:
        if self.lexicon:
            return [lab for _, lab in lexicon_decode(text, probs, self.lexicon, self.unknown_logp, k)]
        return [[int(x > 0.5) for x in probs]]

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
        return [join(t, self.labels(t, p)[0]) + t[len(p) :] for t, p in zip(texts, self.probs(texts))]

    def segment_nbest(self, text: str, k: int) -> list[str]:
        """Top-k segmentations of one text."""
        return self.candidates(text, self.probs([text])[0], k)

    def plausible(self, segmentation: str, min_count: int = 50) -> bool:
        """Every letter token is a common known token of two letters or more with a vowel: used to
        keep alternative segmentations from offering fragments such as `trớ i` or `để p`."""
        if not self.lexicon:
            return True
        for w in segmentation.lower().split():
            if w.isalpha() and (len(w) < 2 or self.lexicon.get(w, 0) < min_count or not any(v in w for v in "aeiouy")):
                return False
        return True

    def candidates(self, text: str, p: list[float], k: int) -> list[str]:
        """Top-k segmentations from given probabilities: lexicon decoding, or a beam over
        independent per-character decisions."""
        if not p:
            return [text]
        if self.lexicon:
            segs = [join(text[: len(p)], lab) + text[len(p) :] for lab in self.labels(text, p, max(4 * k, 16))]
            return list(dict.fromkeys(segs))[:k]
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


class OnnxBoundaryModel(BoundaryModel):
    """The same boundary model exported with `virestore.export_onnx --boundary`, run by onnxruntime
    (no PyTorch at run time). Each text runs at its own length, as OnnxTaggerRestorer does."""

    def __init__(self, path: str, onnx_file: str = "boundary.onnx", threads: int | None = None, batch_size: int = 64,
                 use_lexicon: bool = False, unknown_logp: float = -8.0):
        import onnxruntime as ort

        from .tagger import Vocab

        p = Path(path)
        self.cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        self.vocab = Vocab(self.cfg["vocab"])
        self.max_len, self.batch_size = self.cfg["model"]["max_len"], batch_size
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(p / onnx_file), opts, providers=["CPUExecutionProvider"])
        self.device = "cpu"
        lex = p / "lexicon.json"
        self.lexicon = json.loads(lex.read_text(encoding="utf-8")) if use_lexicon and lex.exists() else None
        self.unknown_logp = unknown_logp
        self.name = f"boundary-onnx:{p.name}/{onnx_file}" + ("+lexicon" if self.lexicon else "")

    def probs(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        out: list[list[float]] = []
        for t in texts:
            t = t[: self.max_len]
            if not t:
                out.append([])
                continue
            ids = np.array([self.vocab.encode(t) + [0] * (len(t) < 2)], dtype=np.int64)  # the graph needs 2 positions
            logits = self.sess.run(["logits"], {"ids": ids})[0][0, : len(t)].astype(np.float64)
            # softmax over two classes: probability of class 1
            out.append((1.0 / (1.0 + np.exp(logits[:, 0] - logits[:, 1]))).tolist())
        return out


class SegmentTagger:
    """Restorer for unspaced input: boundaries first, then the diacritic tagger.

    For evaluation on the usual test files, the spaces in `input` are removed first, so the
    output is scored against the ordinary spaced references.
    """

    def __init__(self, boundary_dir: str, tagger_dir: str, threads: int | None = None, batch_size: int = 64,
                 onnx_file: str | None = None, unknown_logp: float | None = None):
        from .models.tagger import OnnxTaggerRestorer, TaggerRestorer

        self.boundary = BoundaryModel(boundary_dir, threads=threads, batch_size=batch_size,
                                      use_lexicon=unknown_logp is not None, unknown_logp=unknown_logp or -4.0)
        if onnx_file:
            self.tagger = OnnxTaggerRestorer(tagger_dir, threads=threads, batch_size=batch_size, onnx_file=onnx_file)
        else:
            self.tagger = TaggerRestorer(tagger_dir, threads=threads, batch_size=batch_size)
        self.name = f"segment-tagger:{Path(boundary_dir).name}+{Path(tagger_dir).name}"
        self.device = self.boundary.device

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


class ChunkSuggester:
    """Candidates for the chunk an input method is typing: the text since the last space, which may
    hold several words typed without spaces (`homnaytroidepqua`).

    `raw` is what the user actually pressed. The screen can differ because the Telex engine reads
    some letters of an unspaced chunk as marks (`anhsang` shows as `ánhang`), so segmentation works
    on `raw`. If the screen shows one known syllable that Telex marked from the keys (`tieengs` ->
    `tiếng`), the user is typing Telex and no candidate is offered.
    """

    def __init__(self, boundary: "BoundaryModel", restorer, segmentations: int = 3, context_words: int = 6):
        self.boundary, self.restorer, self.segmentations = boundary, restorer, segmentations
        self.context_words = context_words

    def _near(self, context: str) -> str:
        """The last few words of the context: the tagger mainly needs the neighbourhood, and the
        n-best search cost grows with the text length."""
        words = context.split(" ")
        tail = " ".join(words[-(self.context_words + 1) :])
        return tail

    def is_telex_word(self, screen: str, raw: str) -> bool:
        plain = strip_diacritics(screen)
        return (
            plain != screen
            and " " not in screen.strip()
            and plain.lower() != raw.lower()
            and self.boundary.lexicon is not None
            and plain.lower() in self.boundary.lexicon
        )

    def suggest(self, context: str, screen: str, raw: str = "", k: int = 5) -> dict:
        raw = "".join((raw or strip_diacritics(screen)).split())
        if not raw or self.is_telex_word(screen, raw):
            return {"candidates": [], "telex": bool(raw)}
        context = self._near(context)
        segs = self.boundary.segment_nbest(raw, 4 * self.segmentations)
        segs = (segs[:1] + [s for s in segs[1:] if self.boundary.plausible(s)])[: self.segmentations]
        tagger = getattr(self.restorer, "inner", self.restorer)
        if not hasattr(tagger, "_log_probs"):  # not a character tagger: go through its public interface
            texts = [context + seg for seg in segs]
            cands = self.restorer.restore_nbest(texts[:1], k * 2)[0] + self.restorer.restore(texts[1:])
            out = [c[len(context) :] for c in cands if len(c) - len(context) in {len(s) for s in segs}]
            return {"candidates": list(dict.fromkeys(out))[:k], "telex": False}
        return {"candidates": self._tag(tagger, strip_diacritics(context), segs, k), "telex": False}

    @staticmethod
    def _tag(tagger, context: str, segs: list[str], k: int) -> list[str]:
        """One tagger pass over every segmentation (marks removed from the context); top-k tails of
        the best one by beam search over the chunk's characters only, the best tail of the others."""
        from .ime import has_stacked_tones
        from .models.tagger import resolve_stacked_tones, topk
        from .tagger import TAGGABLE, apply_label, decode

        texts = [context + s for s in segs]
        lp, n = tagger._log_probs(texts)
        start = len(context)
        hyps: list[tuple[float, str]] = [(0.0, "")]
        src = texts[0]
        for j in range(start, min(len(src), n)):
            c = src[j]
            if c not in TAGGABLE:
                hyps = [(s, t + c) for s, t in hyps]
                continue
            opts = [(v, apply_label(c, l)) for v, l in topk(lp[0, j], min(3, len(tagger._valid(c))))]
            hyps = heapq.nlargest(max(8 * k, 16), ((s + v, t + ch) for s, t in hyps for v, ch in opts))
        best = [t + src[n:] for _, t in hyps if not has_stacked_tones(t)]
        others = []
        labels = lp.argmax(-1).tolist()
        for b in range(1, len(texts)):
            s = texts[b][:n]
            lab = resolve_stacked_tones(s, labels[b][: len(s)], lp[b], tagger._valid)
            others.append((decode(s, lab) + texts[b][n:])[start:])
        # three readings of the best segmentation, then the other segmentations, then the rest
        return list(dict.fromkeys(best[:3] + others + best[3:]))[:k]


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
        for ids, mask, lab in _batches(rows, vocab, t["batch_size"], m["max_len"], True, rng, d.get("prefix_prob", 0.0),
                                       d.get("span_prob", 0.0)):
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


def error_breakdown(rows: list[tuple[str, list[int]]], preds: list[list[int]], examples: int = 40) -> dict:
    """Boundary errors between two letters (syllable segmentation) vs. next to digits or punctuation.

    `letters_exact` is the share of sentences whose letter-letter boundaries are all right; an
    input method splits typed text at punctuation, so these are the boundaries it has to find.
    """
    counts = {"letters": [0, 0], "other": [0, 0]}  # [boundary positions, errors]
    letters_exact, shown = 0, []
    for (src, gold), pred in zip(rows, preds):
        ok = True
        for i in range(len(src) - 1):
            kind = "letters" if src[i].isalpha() and src[i + 1].isalpha() else "other"
            counts[kind][0] += 1
            if pred[i] != gold[i]:
                counts[kind][1] += 1
                ok = ok and kind != "letters"
        letters_exact += ok
        if not ok and len(shown) < examples:
            shown.append({"gold": join(src, gold), "pred": join(src, pred)})
    return {
        "letters_exact": round(letters_exact / max(len(rows), 1), 6),
        "error_rate_letters": round(counts["letters"][1] / max(counts["letters"][0], 1), 6),
        "error_rate_other": round(counts["other"][1] / max(counts["other"][0], 1), 6),
        "errors_letters": counts["letters"][1],
        "errors_other": counts["other"][1],
        "examples": shown,
    }


def chunk_scores(bm: "BoundaryModel", texts: list[str], sizes: list[int]) -> dict:
    """Exact segmentation of chunks of n consecutive words made only of letters, segmented alone,
    as an input method sees them when the user types a few words before choosing."""
    out = {}
    for n in sizes:
        chunks = []
        for t in texts:
            words = strip_diacritics(t).split()
            for i in range(0, len(words) - n + 1, n):
                group = words[i : i + n]
                if all(w.isalpha() for w in group):
                    chunks.append(" ".join(group))
        pairs = [boundary_pair(c) for c in chunks]
        probs = bm.probs([s for s, _ in pairs])
        hits = {1: 0, 3: 0, 5: 0}
        for (s, _), p, c in zip(pairs, probs, chunks):
            cands = bm.candidates(s, p, 5)
            for k in hits:
                hits[k] += c in cands[:k]
        out[str(n)] = {"chunks": len(chunks), **{f"exact@{k}": round(v / max(len(chunks), 1), 6) for k, v in hits.items()}}
    return out


def evaluate_dir(model_dir: str, test: Path, limit: int | None, tag: str, results: Path,
                 chunk_sizes: list[int] | None = None, use_lexicon: bool = False, unknown_logp: float = -8.0) -> dict:
    rows = load_rows(test, limit)
    bm = BoundaryModel(model_dir, use_lexicon=use_lexicon, unknown_logp=unknown_logp)
    t0 = time.perf_counter()
    preds = [bm.labels(s, p)[0] for (s, _), p in zip(rows, bm.probs([s for s, _ in rows]))]
    scores = boundary_scores(preds, [g for _, g in rows])
    rec = {"tag": tag, "model": bm.name, "test": str(test), "device": bm.device, "boundary": scores,
           "unknown_logp": unknown_logp if bm.lexicon else None,
           "breakdown": error_breakdown(rows, preds), "seconds": round(time.perf_counter() - t0, 2)}
    if chunk_sizes:
        texts = []
        with test.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                texts.append(r.get("input") or r["target"])
        rec["chunks"] = chunk_scores(bm, texts, chunk_sizes)
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
    ap.add_argument("--chunks", help="also score chunks of these word counts, e.g. 1,2,3,4,6,8")
    ap.add_argument("--build-lexicon", type=Path, help="write <--eval dir>/lexicon.json from this training JSONL")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--lexicon", action="store_true", help="decode with <--eval dir>/lexicon.json")
    ap.add_argument("--unknown-logp", type=float, default=-8.0, help="cost of a letter token not in the lexicon")
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    if a.config:
        import yaml

        train(yaml.safe_load(a.config.read_text(encoding="utf-8")))
    elif a.eval and a.build_lexicon:
        lex = build_lexicon(a.build_lexicon, a.min_count)
        (Path(a.eval) / "lexicon.json").write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
        print(f"{len(lex)} tokens with count >= {a.min_count} -> {Path(a.eval) / 'lexicon.json'}")
    elif a.eval and a.test:
        sizes = [int(x) for x in a.chunks.split(",")] if a.chunks else None
        rec = evaluate_dir(a.eval, a.test, a.limit, a.tag or f"boundary-{Path(a.eval).name}", a.results, sizes,
                           use_lexicon=a.lexicon, unknown_logp=a.unknown_logp)
        print(json.dumps({k: v for k, v in rec.items() if k != "breakdown"} | {"breakdown": {k: v for k, v in rec["breakdown"].items() if k != "examples"}}, indent=1))
    else:
        ap.error("give --config to train, or --eval and --test to evaluate")


if __name__ == "__main__":
    main()
