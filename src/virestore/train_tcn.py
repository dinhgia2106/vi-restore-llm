"""Retrain the acausal TCN baseline (E1) on this project's sentence pairs.

The original checkpoint of github.com/dinhgia2106/Vietnamese-diacritic-restoration is lost, so the
model is retrained here with that repo's own code — `VietnameseAccentRestorer(use_enhanced_model=True)`,
`AccentDataset` and `AccentRestorationTrainer` (AdamW, label smoothing, early stopping) — on the clean
training split instead of the Viet74K word list it was first trained on. Only the data, the number of
epochs and the batch size differ from that repo's `train_model.py`.

    python -m virestore.train_tcn --tcn-repo ../Vietnamese-diacritic-restoration \
        --train data/processed/train.jsonl --dev data/processed/dev.jsonl --out runs/tcn

Evaluate with `virestore.eval --model tcn --tcn-repo ... --tcn-weights runs/tcn/best_model.pth`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import utf8_stdio


def load_pairs(path: Path, limit: int | None, max_length: int) -> tuple[list[str], list[str]]:
    inputs, targets = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(inputs) >= limit:
                break
            r = json.loads(line)
            if len(r["input"]) == len(r["target"]) <= max_length:
                inputs.append(r["input"])
                targets.append(r["target"])
    return inputs, targets


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcn-repo", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/tcn"))
    ap.add_argument("--limit", type=int, default=None, help="training sentences (default: all)")
    ap.add_argument("--dev-limit", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=15, help="upper bound; the repo's early stopping applies")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(a.tcn_repo.resolve()))
    from model_architecture import VietnameseAccentRestorer  # type: ignore
    from train_model import AccentDataset, AccentRestorationTrainer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_in, tr_out = load_pairs(a.train, a.limit, a.max_length)
    dv_in, dv_out = load_pairs(a.dev, a.dev_limit, a.max_length)
    print(f"train {len(tr_in)} dev {len(dv_in)} device {device}", flush=True)

    model = VietnameseAccentRestorer(use_enhanced_model=True)
    train_ds = AccentDataset(tr_in, tr_out, model.char_to_idx, a.max_length)
    dev_ds = AccentDataset(dv_in, dv_out, model.char_to_idx, a.max_length)
    kw = dict(batch_size=a.batch_size, num_workers=4, pin_memory=device.type == "cuda")
    trainer = AccentRestorationTrainer(model, device)
    a.out.mkdir(parents=True, exist_ok=True)
    trainer.train(DataLoader(train_ds, shuffle=True, **kw), DataLoader(dev_ds, shuffle=False, **kw), a.epochs, save_dir=str(a.out))
    (a.out / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
