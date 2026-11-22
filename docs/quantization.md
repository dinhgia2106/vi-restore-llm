# Quantisation with llama.cpp

1. Merge the adapter into the base model:

   ```bash
   python -m virestore.merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05-lora --out runs/qwen05-lora-merged
   ```

2. Convert to GGUF and quantise (from a llama.cpp checkout):

   ```bash
   python convert_hf_to_gguf.py ../vi-restore-llm/runs/qwen05-lora-merged --outfile qwen05-lora-f16.gguf --outtype f16
   ./llama-quantize qwen05-lora-f16.gguf ../vi-restore-llm/models/qwen05-lora-q8_0.gguf Q8_0
   ./llama-quantize qwen05-lora-f16.gguf ../vi-restore-llm/models/qwen05-lora-q4_k_m.gguf Q4_K_M
   ```

3. Evaluate every variant with the same test set and latency settings:

   ```bash
   python -m virestore.eval --model gguf --gguf models/qwen05-lora-q8_0.gguf   --test data/processed/test.jsonl --tag e4-q8
   python -m virestore.eval --model gguf --gguf models/qwen05-lora-q4_k_m.gguf --test data/processed/test.jsonl --tag e4-q4km
   ```

Record the CPU model and thread count; latency numbers are only comparable on the same machine.

## What it took on the molab sandbox (E4)

The `e4_setup` / `e4_eval` cells of `notebooks/experiments.py` run the steps above. Things that
were needed beyond the commands:

- `cmake`, `ninja` and `sentencepiece` from PyPI (the sandbox has gcc but no `make`), and
  `cmake -G Ninja`. Only the `llama-quantize` target is built; `llama-cpp-python` builds its own
  copy of llama.cpp when installed.
- The sandbox sets `PYTHONSAFEPATH=1`, so `convert_hf_to_gguf.py` cannot import its own
  `conversion/` package unless the llama.cpp checkout is on `PYTHONPATH`
  (`PYTHONPATH=<llama.cpp>:<llama.cpp>/gguf-py`).
- The Qwen2 converter tries the SentencePiece vocab first and falls back to BPE only on
  `FileNotFoundError`; without `sentencepiece` installed it fails with `ModuleNotFoundError`.
- `merge.py` merges in float16 and the converter writes `--outtype f16`; the f16 GGUF is evaluated
  too, as the unquantised llama.cpp reference.

CPU decoding through llama.cpp is slow on the sandbox (about 0.3 s per sentence for Q4_K_M), so E4
is evaluated on fixed subsets (first 2 000 clean test sentences, first 1 000 hard sentences), and
the HF model from E3 is re-run on the same subsets (`e4-ref-hf-*`). The GGUF files are in the
private HF repo `dinhgia2106/vi-restore-llm-qwen05-lora-gguf`.

## E4 results

From `results/e4-*.json` (clean2k = first 2 000 clean test sentences, hard1k = first 1 000 hard
sentences). Latency is single-sentence on the **sandbox CPU** (20 vCPUs, gVisor), not the target
laptop.

| run | file size | clean2k CER | clean2k sent acc | hard1k CER | hard1k sent acc | p95 ms (clean2k, sandbox CPU) |
|---|---|---|---|---|---|---|
| HF + adapter (PyTorch, GPU) | – | 0.0121 | 0.5440 | 0.0180 | 0.4830 | – |
| GGUF f16 | 994 MB | 0.0105 | 0.5910 | 0.0158 | 0.5240 | 737.43 |
| GGUF Q8_0 | 531 MB | 0.0105 | 0.5895 | 0.0158 | 0.5240 | 520.44 |
| GGUF Q4_K_M | 398 MB | 0.0108 | 0.5780 | 0.0164 | 0.4960 | 489.83 |

- **Quantisation cost** (against f16 GGUF, same runtime): Q8_0 is essentially free; Q4_K_M costs
  0.0003 CER / 1.3 points sentence accuracy on clean and 0.0006 CER / 2.8 points on hard.
- **Latency**: even Q4_K_M has a p95 near 490 ms on the sandbox CPU, about ten times the 50 ms
  budget. The laptop number is still to be measured.
- **Open question — HF vs llama.cpp.** The same merged model scores better through llama.cpp
  than through PyTorch (clean2k CER 0.0105 vs 0.0121; 505 of 2 000 outputs differ, almost all in
  diacritic choices, not length). A diagnostic on 300 sentences ruled out the obvious causes:
  bf16 vs fp32 weights, merged vs unmerged adapter, and batch size 64 vs 1 all give CER
  0.0106–0.0118 against 0.0098 for f16 GGUF, and llama.cpp decodes greedily
  (`temperature=0`, `repeat_penalty=1.0`). Remaining suspects: how llama.cpp renders the chat
  template into the prompt, and numerical differences between the kernels. Until this is
  resolved, compare quantised variants against the f16 GGUF, not against the HF run.
