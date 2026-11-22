# Results

| run | model | test set | n | CER ↓ | WER ↓ | word acc ↑ | sentence acc ↑ | p50 ms | p95 ms | RSS MB | device |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e0-identity-clean | identity | processed/test | 19956 | 0.2111 | 0.8284 | 0.1716 | 0.0000 | 0.0 | 0.0 | 45.1 | cpu |
| e0-identity-hard | identity | processed/test_hard | 2801 | 0.2218 | 0.8450 | 0.1550 | 0.0000 | 0.0 | 0.0 | 31.4 | cpu |
| e0-identity-noisy | identity | processed_noisy/test | 19956 | 0.2271 | 0.8376 | 0.1624 | 0.0000 | 0.0 | 0.0 | 44.7 | cpu |
| e0-lexicon-clean | lexicon | processed/test | 19956 | 0.0342 | 0.1426 | 0.8573 | 0.1576 | 0.05 | 0.08 | 1897.6 | cpu |
| e0-lexicon-hard | lexicon | processed/test_hard | 2801 | 0.0459 | 0.1879 | 0.8121 | 0.1317 | 0.03 | 0.07 | 1881.4 | cpu |
| e0-lexicon-noisy | lexicon | processed_noisy/test | 19956 | 0.0630 | 0.2019 | 0.7981 | 0.0724 | 0.05 | 0.08 | 1897.7 | cpu |
| e2-qwen05-zeroshot-clean | hf:Qwen/Qwen2.5-0.5B-Instruct | processed/test | 1000 | 0.5687 | 1.0144 | 0.1453 | 0.0000 | 325.01 | 666.6 | 3311.7 | cuda |
| e3-qwen05-lora-clean | hf:Qwen/Qwen2.5-0.5B-Instruct+/marimo/vi-restore-llm/runs/qwen05-lora | processed/test | 19956 | 0.0129 | 0.0493 | 0.9520 | 0.5219 | 290.5 | 487.97 | 3352.6 | cuda |
| e3-qwen05-lora-hard | hf:Qwen/Qwen2.5-0.5B-Instruct+/marimo/vi-restore-llm/runs/qwen05-lora | processed/test_hard | 2801 | 0.0181 | 0.0698 | 0.9315 | 0.4681 | 259.9 | 476.37 | 3315.9 | cuda |
| e3-qwen05-lora-noisy | hf:Qwen/Qwen2.5-0.5B-Instruct+/marimo/vi-restore-llm/runs/qwen05-lora | processed_noisy/test | 19956 | 0.0428 | 0.1172 | 0.8871 | 0.1919 | 310.11 | 512.76 | 3358.4 | cuda |
| e4-gguf-f16-clean2k | gguf:qwen05-lora-f16.gguf | processed/test | 2000 | 0.0105 | 0.0406 | 0.9603 | 0.5910 | 441.93 | 737.43 | 1420.8 | cpu |
| e4-gguf-f16-hard1k | gguf:qwen05-lora-f16.gguf | processed/test_hard | 1000 | 0.0158 | 0.0615 | 0.9398 | 0.5240 | 408.8 | 723.69 | 1434.9 | cpu |
| e4-gguf-q4_k_m-clean2k | gguf:qwen05-lora-q4_k_m.gguf | processed/test | 2000 | 0.0108 | 0.0423 | 0.9583 | 0.5780 | 303.0 | 489.83 | 897.4 | cpu |
| e4-gguf-q4_k_m-hard1k | gguf:qwen05-lora-q4_k_m.gguf | processed/test_hard | 1000 | 0.0164 | 0.0637 | 0.9369 | 0.4960 | 270.91 | 476.79 | 897.1 | cpu |
| e4-gguf-q8_0-clean2k | gguf:qwen05-lora-q8_0.gguf | processed/test | 2000 | 0.0105 | 0.0411 | 0.9596 | 0.5895 | 311.42 | 520.44 | 992.6 | cpu |
| e4-gguf-q8_0-hard1k | gguf:qwen05-lora-q8_0.gguf | processed/test_hard | 1000 | 0.0158 | 0.0616 | 0.9391 | 0.5240 | 286.93 | 518.15 | 971.5 | cpu |
| e4-ref-hf-clean2k | hf:Qwen/Qwen2.5-0.5B-Instruct+/marimo/vi-restore-llm/runs/qwen05-lora | processed/test | 2000 | 0.0121 | 0.0472 | 0.9532 | 0.5440 | – | – | 3304.1 | cuda |
| e4-ref-hf-hard1k | hf:Qwen/Qwen2.5-0.5B-Instruct+/marimo/vi-restore-llm/runs/qwen05-lora | processed/test_hard | 1000 | 0.0180 | 0.0697 | 0.9315 | 0.4830 | – | – | 3286.2 | cuda |
