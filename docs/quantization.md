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
