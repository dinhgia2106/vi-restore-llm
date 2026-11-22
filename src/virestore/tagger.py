"""Character-level diacritic tagger: one forward pass per sentence, no generation.

Stripping diacritics keeps the string length (one character in, one character out), so clean
restoration is a tagging problem: for every vowel (and ``d``) of the input, predict which of 30
labels to apply — a letter modifier (none / breve ă / circumflex â ê ô / horn ơ ư / stroke đ) times
one of six tones. Other characters are copied. The output therefore always keeps the input's
letters, and inference costs a single encoder pass instead of one decoder pass per token.

    python -m virestore.tagger --config configs/tagger.yaml

The model is a small Transformer encoder over characters. It does not handle noisy input (Telex
leftovers, typos), which needs letters to change; that remains a sequence-to-sequence task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path

from . import utf8_stdio

# combining marks, NFD
BREVE, CIRCUMFLEX, HORN = "̆", "̂", "̛"
TONES = ("", "̀", "́", "̉", "̃", "̣")  # none, grave, acute, hook, tilde, dot below
MODIFIERS = ("", BREVE, CIRCUMFLEX, HORN, "stroke")
N_LABELS = len(MODIFIERS) * len(TONES)  # 30
ALLOWED_MODIFIERS = {"a": {0, 1, 2}, "e": {0, 2}, "i": {0}, "o": {0, 2, 3}, "u": {0, 3}, "y": {0}, "d": {0, 4}}
TAGGABLE = set(ALLOWED_MODIFIERS) | {c.upper() for c in ALLOWED_MODIFIERS}


def char_label(ch: str) -> int:
    """Label of an accented character: modifier index * 6 + tone index (0 for anything plain)."""
    if ch in "đĐ":
        return 4 * len(TONES)
    marks = unicodedata.normalize("NFD", ch)[1:]
    mod = next((i for i, m in enumerate(MODIFIERS[1:4], 1) if m in marks), 0)
    tone = next((i for i, t in enumerate(TONES[1:], 1) if t in marks), 0)
    return mod * len(TONES) + tone


def apply_label(base: str, label: int) -> str:
    """Put label's marks on a plain character; marks that the letter cannot take are dropped."""
    if base not in TAGGABLE:
        return base
    mod, tone = divmod(label, len(TONES))
    low = base.lower()
    if mod not in ALLOWED_MODIFIERS[low]:
        mod = 0
    if low == "d":
        return ("Đ" if base.isupper() else "đ") if mod == 4 else base
    return unicodedata.normalize("NFC", base + MODIFIERS[mod] + TONES[tone])


def labels_for(target: str) -> list[int]:
    return [char_label(c) for c in unicodedata.normalize("NFC", target)]


def decode(source: str, labels: list[int]) -> str:
    return "".join(apply_label(c, l) for c, l in zip(source, labels))


# --------------------------------------------------------------------------- model


def build_model(vocab_size: int, d_model: int, layers: int, heads: int, ff: int, max_len: int, dropout: float):
    import torch
    from torch import nn

    class CharTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, N_LABELS)

        def forward(self, ids, mask):
            pos = torch.arange(ids.shape[1], device=ids.device)
            h = self.embed(ids) + self.pos(pos)[None]
            h = self.encoder(h, src_key_padding_mask=~mask)
            return self.head(self.norm(h))

    return CharTagger()


class Vocab:
    """Input characters; 0 = padding, 1 = unknown."""

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.index = {c: i + 2 for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.index.get(c, 1) for c in text]

    def __len__(self) -> int:
        return len(self.chars) + 2


def _batches(rows, vocab, batch_size, max_len, shuffle, rng):
    import torch

    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [rows[j] for j in order[i : i + batch_size]]
        n = min(max(len(r[0]) for r in chunk), max_len)
        ids = torch.zeros((len(chunk), n), dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        for k, (src, labels) in enumerate(chunk):
            src, labels = src[:n], labels[:n]
            ids[k, : len(src)] = torch.tensor(vocab.encode(src))
            lab[k, : len(src)] = torch.tensor([l if c in TAGGABLE else -100 for c, l in zip(src, labels)])
        yield ids, ids != 0, lab


def load_pairs(path: Path, limit: int | None = None) -> list[tuple[str, list[int]]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            src, tgt = r["input"], unicodedata.normalize("NFC", r["target"])
            if len(src) == len(tgt):  # clean pairs always are; skip the rare odd one
                rows.append((src, labels_for(tgt)))
    return rows


def save(model, vocab: Vocab, cfg: dict, out: Path) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps({**cfg, "vocab": vocab.chars}, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path, device: str = "cpu"):
    import torch

    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    m = cfg["model"]
    model = build_model(len(cfg["vocab"]) + 2, m["d_model"], m["layers"], m["heads"], m["ff"], m["max_len"], 0.0)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    return model.to(device).eval(), Vocab(cfg["vocab"]), cfg


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    rng = random.Random(cfg.get("seed", 13))
    torch.manual_seed(cfg.get("seed", 13))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_pairs(Path(d["train"]), d.get("train_limit"))
    dev = load_pairs(Path(d["dev"]), d.get("dev_limit", 5000))
    counts: dict[str, int] = {}
    for src, _ in train[:500_000]:
        for c in src:
            counts[c] = counts.get(c, 0) + 1
    vocab = Vocab(sorted(c for c, n in counts.items() if n >= d.get("min_char_count", 20)))
    print(f"train {len(train)} dev {len(dev)} vocab {len(vocab)} device {device}", flush=True)

    model = build_model(len(vocab), m["d_model"], m["layers"], m["heads"], m["ff"], m["max_len"], m["dropout"]).to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    steps = t["epochs"] * math.ceil(len(train) / t["batch_size"])
    warm = int(steps * t.get("warmup_ratio", 0.02))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    out = Path(cfg["output_dir"])
    best, step, t0 = -1.0, 0, time.time()

    def evaluate():
        model.eval()
        ok = tot = 0
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for ids, mask, lab in _batches(dev, vocab, 512, m["max_len"], False, rng):
                ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
                pred = model(ids, mask).argmax(-1)
                keep = lab != -100
                ok += (pred[keep] == lab[keep]).sum().item()
                tot += keep.sum().item()
        model.train()
        return ok / max(tot, 1)

    model.train()
    for epoch in range(t["epochs"]):
        for ids, mask, lab in _batches(train, vocab, t["batch_size"], m["max_len"], True, rng):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(ids, mask)
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, N_LABELS), lab.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % t["log_steps"] == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % t["eval_steps"] == 0 or step == steps:
                acc = evaluate()
                print(f"  dev label accuracy {acc:.5f}", flush=True)
                if acc > best:
                    best = acc
                    save(model, vocab, cfg, out)
    print(f"best dev label accuracy {best:.5f}; saved to {out}", flush=True)


if __name__ == "__main__":
    main()
