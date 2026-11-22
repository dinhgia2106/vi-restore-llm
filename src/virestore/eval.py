"""Evaluate any restorer on a JSONL test set: quality + latency + memory.

Examples:
    python -m virestore.eval --model identity --test data/processed/test.jsonl
    python -m virestore.eval --model lexicon --train data/processed/train.jsonl --test data/processed/test.jsonl
    python -m virestore.eval --model hf --hf-model Qwen/Qwen2.5-0.5B-Instruct --test data/processed/test.jsonl --limit 500
    python -m virestore.eval --model hf --hf-model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05-lora --test ...
    python -m virestore.eval --model gguf --gguf models/qwen05-lora-q4_k_m.gguf --test ...

Each run writes results/<tag>.json and refreshes results/summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from . import models, utf8_stdio
from .data import load_jsonl
from .metrics import em_at_k, score


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def machine_info(restorer) -> dict:
    """Latency is only comparable on the same machine, so record where it was measured."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "device": str(getattr(restorer, "device", "cpu")),
        "threads": getattr(restorer, "n_threads", None),
    }


def measure_latency(restorer, inputs: list[str], n: int) -> dict:
    """Single-sentence latency, the way an IME calls the model while the user types."""
    sample = inputs[:n]
    if not sample:
        return {}
    restorer.restore(sample[:1])  # warm-up
    times = []
    for s in sample:
        t0 = time.perf_counter()
        restorer.restore([s])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "n": len(times),
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 2),
        "mean_ms": round(statistics.fmean(times), 2),
    }


def write_summary(results_dir: Path) -> Path:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.endswith((".analysis.json", ".profile.json")):  # analysis / latency profiles, not eval runs
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        q, lat = r["quality"], r.get("latency", {})
        test = Path(r["test_file"])
        em = " / ".join(f"{q[f'em@{k}']:.4f}" if f"em@{k}" in q else "–" for k in (1, 3, 5))
        rows.append(
            f"| {r['tag']} | {r['model']} | {test.parent.name}/{test.stem} | {q['n']} | {q['cer']:.4f} | {q['wer']:.4f} | {q['word_acc']:.4f} | "
            f"{q['sent_acc']:.4f} | {em} | {lat.get('p50_ms', '–')} | {lat.get('p95_ms', '–')} | {r.get('rss_mb') or '–'} | "
            f"{r['machine'].get('device', '–')} |"
        )
    header = (
        "| run | model | test set | n | CER ↓ | WER ↓ | word acc ↑ | sentence acc ↑ | EM@1 / @3 / @5 ↑ | p50 ms | p95 ms | RSS MB | device |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    out = results_dir / "summary.md"
    out.write_text("# Results\n\n" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return out


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["identity", "lexicon", "tcn", "hf", "gguf", "tagger", "tagger-onnx"])
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--train", type=Path, help="training JSONL (lexicon baseline)")
    ap.add_argument("--hf-model", help="Hugging Face model id or path")
    ap.add_argument("--adapter", help="LoRA adapter directory")
    ap.add_argument("--gguf", help="path to a .gguf file")
    ap.add_argument("--tagger", help="directory of a trained virestore.tagger model")
    ap.add_argument("--tcn-repo", help="path to the Vietnamese-diacritic-restoration repo")
    ap.add_argument("--tcn-weights", help="TCN checkpoint (.pth)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for GGUF models (default: all)")
    ap.add_argument("--draft", choices=["none", "lexicon"], default="none", help="GGUF speculative decoding draft (lexicon needs --train)")
    ap.add_argument("--draft-tokens", type=int, default=32, help="max draft tokens proposed per step")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on the first N test rows")
    ap.add_argument("--latency-n", type=int, default=100, help="sentences used for latency (0 = skip)")
    ap.add_argument("--tag", help="name of this run (default: derived from model)")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument(
        "--nbest", type=int, default=0,
        help="also produce K candidates per sentence and report EM@1/3/5 (k <= K); 0 = off",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.test, args.limit)
    inputs = [r["input"] for r in rows]
    refs = [r["target"] for r in rows]

    restorer = models.load(
        args.model,
        train=args.train,
        model=args.hf_model or args.gguf or args.tagger,
        adapter=args.adapter,
        batch_size=args.batch_size,
        threads=args.threads,
        draft=args.draft,
        draft_tokens=args.draft_tokens,
        tcn_repo=args.tcn_repo,
        tcn_weights=args.tcn_weights,
    )

    t0 = time.perf_counter()
    preds = restorer.restore(inputs)
    wall = time.perf_counter() - t0
    decode_stats = restorer.stats() if hasattr(restorer, "stats") else None  # quality pass only
    quality = score(preds, refs).as_dict()
    nbest = None
    if args.nbest:
        if not hasattr(restorer, "restore_nbest"):
            raise SystemExit(f"{restorer.name} cannot produce n-best candidates")
        # EM@1 comes from the n-best search (beam search for LLMs) and can differ from sent_acc,
        # which scores the default (greedy) output.
        nbest = restorer.restore_nbest(inputs, args.nbest)
        quality.update(em_at_k(nbest, refs, tuple(k for k in (1, 3, 5) if k <= args.nbest)))
    latency = measure_latency(restorer, inputs, args.latency_n) if args.latency_n else {}

    tag = args.tag or restorer.name.replace("/", "_").replace(":", "-")
    result = {
        "tag": tag,
        "model": restorer.name,
        "test_file": str(args.test),
        "quality": quality,
        "latency": latency,
        "throughput_sent_per_s": round(len(inputs) / wall, 2) if wall else None,
        "rss_mb": round(_rss_mb(), 1) if _rss_mb() else None,
        "machine": machine_info(restorer),
        **({"decode_stats": decode_stats} if decode_stats else {}),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_predictions:
        with (args.results / f"{tag}.predictions.jsonl").open("w", encoding="utf-8") as fh:
            for j, (i, p, r) in enumerate(zip(inputs, preds, refs)):
                row = {"input": i, "pred": p, "target": r}
                if nbest is not None:
                    row["nbest"] = nbest[j]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = write_summary(args.results)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()
