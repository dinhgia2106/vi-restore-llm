"""Error analysis of a predictions file written by ``virestore.eval --save-predictions``.

    python -m virestore.analysis results/e3-qwen05-lora-clean.predictions.jsonl [more files ...]

For each file it reports:

- **base-letter fidelity**: share of outputs whose letters, once diacritics are removed, equal the
  input's — i.e. the model only added marks and did not rewrite, drop or invent words. Only
  meaningful for the clean task (noisy inputs need letters to change);
- **word-count changes**: outputs with more or fewer words than the reference;
- **syllable confusions**: on outputs with the reference word count, the most frequent
  (reference -> predicted) syllable substitutions, split into "same base letters" (a wrong
  diacritic choice) and "different base letters" (a rewritten word);
- **CER by sentence length** (in words).

Writes ``<file>.analysis.json`` next to each input and prints a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import utf8_stdio
from .data import load_jsonl
from .metrics import levenshtein
from .text import normalize, strip_diacritics

LENGTH_BUCKETS = ((1, 10), (11, 20), (21, 30), (31, 10_000))


def _base(text: str) -> str:
    return strip_diacritics(normalize(text)).lower()


def analyse(rows: list[dict], top: int = 20) -> dict:
    fidelity = more = fewer = 0
    diacritic_conf: Counter = Counter()
    rewrite_conf: Counter = Counter()
    bucket_err = {b: [0, 0, 0] for b in LENGTH_BUCKETS}  # char errors, ref chars, sentences
    for r in rows:
        pred, ref, src = normalize(r["pred"]), normalize(r["target"]), r["input"]
        fidelity += _base(pred) == _base(src)
        pw, rw = pred.split(), ref.split()
        more += len(pw) > len(rw)
        fewer += len(pw) < len(rw)
        if len(pw) == len(rw):
            for p, t in zip(pw, rw):
                if p != t:
                    (diacritic_conf if _base(p) == _base(t) else rewrite_conf)[(t, p)] += 1
        for lo, hi in LENGTH_BUCKETS:
            if lo <= len(rw) <= hi:
                e = bucket_err[(lo, hi)]
                e[0] += levenshtein(pred, ref)
                e[1] += max(len(ref), 1)
                e[2] += 1
                break
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "base_letter_fidelity": round(fidelity / n, 5),
        "more_words": round(more / n, 5),
        "fewer_words": round(fewer / n, 5),
        "diacritic_errors": sum(diacritic_conf.values()),
        "rewrite_errors": sum(rewrite_conf.values()),
        "top_diacritic_confusions": [[t, p, c] for (t, p), c in diacritic_conf.most_common(top)],
        "top_rewrites": [[t, p, c] for (t, p), c in rewrite_conf.most_common(top)],
        "cer_by_length": {
            f"{lo}-{hi if hi < 10_000 else ''}": {"sentences": s, "cer": round(err / tot, 5) if tot else None}
            for (lo, hi), (err, tot, s) in bucket_err.items()
        },
    }


def to_markdown(name: str, a: dict, show: int = 10) -> str:
    lines = [
        f"### {name}",
        "",
        f"- sentences: {a['n']}",
        f"- base-letter fidelity: {a['base_letter_fidelity']:.4f}",
        f"- more / fewer words than the reference: {a['more_words']:.4f} / {a['fewer_words']:.4f}",
        f"- word errors on same-length outputs: {a['diacritic_errors']} wrong diacritics, {a['rewrite_errors']} rewritten words",
        "",
        "| length (words) | sentences | CER |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['sentences']} | {v['cer'] if v['cer'] is not None else '–'} |" for k, v in a["cer_by_length"].items()]
    lines += ["", "Top wrong diacritics (reference → predicted, count): " + ", ".join(f"{t} → {p} ({c})" for t, p, c in a["top_diacritic_confusions"][:show])]
    lines += ["", "Top rewrites: " + ", ".join(f"{t} → {p} ({c})" for t, p, c in a["top_rewrites"][:show]), ""]
    return "\n".join(lines)


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    for path in args.predictions:
        a = analyse(load_jsonl(path), args.top)
        out = path.with_name(path.name.replace(".predictions.jsonl", "") + ".analysis.json")
        out.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
        print(to_markdown(path.name.replace(".predictions.jsonl", ""), a))


if __name__ == "__main__":
    main()
