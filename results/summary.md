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
