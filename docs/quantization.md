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
