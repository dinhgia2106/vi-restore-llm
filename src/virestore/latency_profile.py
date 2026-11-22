"""Where does a GGUF restoration spend its time on this CPU?

    python -m virestore.latency_profile --gguf models/qwen05-lora-q4_k_m.gguf --test data/processed/test.jsonl \
        --threads 8 --tag q4_k_m-t8

Measures, single sentence at a time (the way an IME calls the model):

- the full restoration (prompt + generated sentence) and the prompt alone (``max_tokens=1``);
- the cost of one forward pass over n new tokens at once (n = 1, 2, 4, 8, 16, 32), which is what
  speculative decoding pays to verify an n-token draft.

Writes ``results/profile-<tag>.profile.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from . import utf8_stdio
from .data import load_jsonl
from .eval import machine_info
from .prompts import messages


def _median_ms(fn, items) -> float:
    ts = []
    for x in items:
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 2)


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="sentences")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=20, help="repeats per batch size for the forward-pass costs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()

    from llama_cpp import Llama

    from .models.gguf import GGUFRestorer

    rows = load_jsonl(a.test, a.n)
    inputs = [r["input"] for r in rows]
    restorer = GGUFRestorer(a.gguf, n_threads=a.threads)
    llm: Llama = restorer.llm

    def chat(x, max_tokens):
        return llm.create_chat_completion(messages=messages(x), temperature=0.0, max_tokens=max_tokens)

    chat(inputs[0], 4)  # warm-up
    usage = [chat(x, len(x) + 32)["usage"] for x in inputs[:20]]
    full_ms = _median_ms(lambda x: chat(x, len(x) + 32), inputs)
    prompt_ms = _median_ms(lambda x: chat(x, 1), inputs)

    toks = llm.tokenize(rows[0]["target"].encode("utf-8"), add_bos=False)
    while len(toks) < 32:
        toks = toks + toks
    prefix = llm.tokenize("Xin chào".encode("utf-8"), add_bos=False)  # some context before the measured pass
    forward = {}
    for n in (1, 2, 4, 8, 16, 32):
        ts = []
        for _ in range(a.repeats):
            llm.reset()
            llm.eval(prefix)
            t = time.perf_counter()
            llm.eval(toks[:n])
            ts.append((time.perf_counter() - t) * 1000)
        forward[str(n)] = round(statistics.median(ts), 2)

    result = {
        "tag": a.tag,
        "model": restorer.name,
        "test_file": str(a.test),
        "sentences": len(inputs),
        "median_prompt_tokens": statistics.median(u["prompt_tokens"] for u in usage),
        "median_completion_tokens": statistics.median(u["completion_tokens"] for u in usage),
        "median_full_ms": full_ms,
        "median_prompt_plus_one_token_ms": prompt_ms,
        "forward_ms_by_new_tokens": forward,
        "machine": machine_info(restorer),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a.results.mkdir(parents=True, exist_ok=True)
    out = a.results / f"profile-{a.tag}.profile.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
