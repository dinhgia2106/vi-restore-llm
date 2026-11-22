# Experiment plan

## Question

Under a 50 ms p95 single-sentence latency budget on a laptop CPU, which approach restores Vietnamese diacritics best, and what does on-device deployment cost in quality?

## Hypotheses

- **H1** — A LoRA-tuned 0.5B LLM beats the acausal TCN on sentence exact match, mainly on ambiguous syllables (`ma / mà / má / mả / mã / mạ`) where wider context matters.
- **H2** — The TCN stays faster; the LLM only fits the 50 ms budget after int4 quantisation and with short inputs.
- **H3** — int4 (GGUF Q4_K_M) loses less than 1 point of sentence accuracy versus the fp16 adapter.
- **H4** — On noisy input (Telex leftovers, typos) the LLM degrades less than the TCN, which was trained only on clean stripping.

## Data

- Corpus: [WanJuan-Vietnamese](https://opendatalab.com/OpenDataLab/WanJuan-Vietnamese) (CC BY 4.0, 74 B tokens). A 2 M-sentence slice is enough for E3–E5; E6 scales it. Source, licence and changes are documented in [data/README.md](../data/README.md).
- Report results on a test set drawn from the same corpus, plus the hard set; the TCN was trained on different data, so also note its training corpus when comparing.
- `data.py` keeps sentences of 3–40 words that contain at least one diacritic, de-duplicates them, and splits 98 / 1 / 1 by content hash.
- Two variants: **clean** (pure stripping) and **noisy** (`NoiseConfig` defaults: 3 % Telex leftovers, 1 % each of dropped, doubled and swapped letters).
- A small **hard set**, reported separately: clean test sentences in which at least 30 % of syllables (and at least two) are ambiguous. A syllable is ambiguous when it occurs ≥ 50 times in training and its most frequent accented form covers ≤ 50 % of those occurrences (`do` → đó / độ / do / đồ). Counting syllables with "many accented forms" does not work at this corpus size: almost every syllable has several, and that rule kept 99.8 % of the test split.

## Matrix

| ID | Model | Train data | Eval sets | Notes |
|---|---|---|---|---|
| E0 | identity, lexicon | – / train | clean, noisy, hard | floors |
| E1 | TCN (existing repo) | its own | clean, noisy, hard | baseline to beat |
| E2 | Qwen2.5-0.5B zero-shot | – | clean (1k) | shows why fine-tuning is needed |
| E3 | Qwen2.5-0.5B + LoRA (`configs/qwen05_lora.yaml`) | clean 200k | clean, noisy, hard | main model |
| E4 | E3 merged → GGUF Q8_0 / Q4_K_M | – | clean, hard | quantisation cost |
| E5 | Qwen2.5-1.5B + QLoRA (`configs/qwen15_qlora.yaml`) | noisy 200k | clean, noisy, hard | quality ceiling for small models |
| E6 | best of E3/E5 with 50k / 200k / 1M examples | – | clean | data-scaling curve |

Report for each: CER, WER, word accuracy, sentence accuracy, p50 / p95 latency, RSS, and machine spec. Fix seeds; run each fine-tune at least twice if variance looks large.

## Error analysis

- Confusion table for the most frequent wrong syllables.
- Sentences where the LLM changes words instead of only adding marks (hallucination rate: output with a different no-diacritic form than the input).
- Latency vs. input length.

## Milestones

| Week | Deliverable |
|---|---|
| 1 | Corpus collected, clean + noisy + hard sets built, E0–E1 numbers |
| 2 | E2–E3, first fine-tuned adapter, error analysis v1 |
| 3 | E4 quantisation + local server, VIMEK integration prototype |
| 4 | E5–E6, final tables and plots, model card on Hugging Face, Spaces demo |
| 5–6 | Write-up (README results, blog post / preprint) |
