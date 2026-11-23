# Vietnamese diacritic restoration under a 50 ms CPU budget

*Research report for vi-restore-llm. Every number below comes from a run saved in `results/`; the
file is named next to each table.*

## Question

An input method (VIMEK) has to restore Vietnamese diacritics while the user types
(`hom nay troi dep qua` → `hôm nay trời đẹp quá`), on the user's CPU, within about **50 ms at the
95th percentile per sentence**. Which approach gives the best restoration under that budget — a
frequency lexicon, an acausal TCN, a LoRA-tuned small LLM, or a dedicated model — and what does
quantising the LLM cost?

## Setup

- **Data.** WanJuan-Vietnamese (CC BY 4.0), 7 shards, one per domain. 2M unique sentences of 3–40
  words: 1 960 206 train / 19 838 dev / 19 956 test, split by sentence hash (`data/README.md`).
- **Test sets.** *clean*: diacritics stripped. *noisy*: plus synthetic Telex leftovers and typos.
  *hard*: 2 801 clean test sentences in which ≥ 30 % of syllables are ambiguous (the syllable's
  most frequent accented form covers ≤ 50 % of its uses in training).
- **Metrics.** CER, WER, word accuracy, sentence exact match (of the default output), EM@k (the
  reference among the top-k candidates, as in a candidate bar), single-sentence p50/p95 latency.
- **Hardware.** Training and quality runs on a molab sandbox GPU. **Latency is measured only on the
  target laptop** (Windows 11, Intel CPU, 32 logical processors), batch size 1.

## Systems

| system | what it is | training data |
|---|---|---|
| identity | input unchanged | – |
| lexicon | most frequent accented form per syllable, bigram tie-breaking | 1.96M pairs |
| Qwen2.5-0.5B zero-shot | instruct model with a restoration prompt | – |
| Qwen2.5-0.5B + LoRA (E3) | LoRA r = 16 on all projections, loss on the answer | 200k clean pairs, 1 epoch |
| … GGUF f16 / Q8_0 / Q4_K_M (E4) | E3 merged and quantised with llama.cpp | – |
| Qwen2.5-1.5B + LoRA on noisy pairs (E5) | LoRA r = 32, bf16 base | 200k noisy pairs, 1 epoch |
| acausal TCN (E1) | the user's TCN, retrained with its repo's code | 300k clean pairs, 3 epochs |
| character tagger (T2) | 14.4M-parameter Transformer encoder over characters; labels every vowel and `d` with one of 30 diacritic labels in one pass | 1.96M clean pairs, 3 epochs |

## Results

### Quality (full test sets)

From `results/summary.md` (E0, E2, E3, T2 runs):

| system | clean CER | clean sent acc | hard CER | hard sent acc | noisy CER | noisy sent acc |
|---|---|---|---|---|---|---|
| identity | 0.2111 | 0.0000 | 0.2218 | 0.0000 | 0.2271 | 0.0000 |
| lexicon | 0.0342 | 0.1576 | 0.0459 | 0.1317 | 0.0630 | 0.0724 |
| acausal TCN (300k pairs) | 0.0291 | 0.1404 | 0.0372 | 0.1185 | 0.0572 | 0.0639 |
| Qwen2.5-0.5B zero-shot (1k clean) | 0.5687 | 0.0000 | – | – | – | – |
| Qwen2.5-0.5B + LoRA | 0.0129 | 0.5219 | 0.0181 | 0.4681 | 0.0428 | 0.1919 |
| Qwen2.5-1.5B + LoRA, noisy training | 0.0121 | 0.5675 | 0.0173 | 0.5223 | **0.0157** | **0.5059** |
| character tagger | 0.0080 | 0.6022 | 0.0114 | 0.5587 | 0.0358 | 0.2097 |
| large character tagger | **0.0058** | **0.6857** | **0.0083** | **0.6458** | 0.0339 | 0.2351 |

The lexicon on the noisy set is trained on noisy pairs, and so is the 1.5B model; the 0.5B LoRA
model and the tagger were trained on clean pairs only.

### Data efficiency (tagger trained on less data)

Same tagger, trained on the first N clean pairs (`results/t6-tagger-*.json`, full test sets):

| training pairs × epochs | clean CER | clean sent acc | hard CER | hard sent acc |
|---|---|---|---|---|
| 50k × 3 | 0.0895 | 0.0087 | 0.1027 | 0.0039 |
| 200k × 1 (the LoRA run's budget) | 0.0698 | 0.0196 | 0.0840 | 0.0118 |
| 200k × 3 | 0.0311 | 0.1544 | 0.0405 | 0.1157 |
| 1M × 3 | 0.0111 | 0.5012 | 0.0153 | 0.4538 |
| 1.96M × 3 | 0.0080 | 0.6022 | 0.0114 | 0.5587 |

At the LoRA run's data budget (200k pairs, one epoch) the tagger is worse than the lexicon, while
the LoRA model reaches CER 0.0129: the pretrained LLM is far more sample-efficient. The tagger
draws level at about 1M pairs (lower CER, slightly lower sentence accuracy than the LoRA model's
0.0129 / 0.5219) and passes it with 1.96M.

### Candidate bar (EM@k, full test sets)

| system | clean @1 / @3 / @5 | hard @1 / @3 / @5 |
|---|---|---|
| lexicon | 0.1824 / 0.2970 / 0.3513 | 0.1653 / 0.2699 / 0.3210 |
| Qwen2.5-0.5B + LoRA (beam 5) | 0.5981 / 0.7397 / 0.7757 | 0.5491 / 0.6951 / 0.7351 |
| character tagger | 0.6022 / 0.7324 / 0.7719 | 0.5587 / 0.6944 / 0.7283 |

With five candidates the correct sentence is offered for about three quarters of clean sentences,
by both the LLM and the tagger, against one third for the lexicon.

### Quantisation (E4; first 2 000 clean / 1 000 hard sentences)

`docs/quantization.md`. Against the f16 GGUF of the same model: Q8_0 is free (clean CER 0.0105 vs
0.0105); Q4_K_M costs 0.0003 CER and 1.3 points of sentence accuracy on clean, 2.8 points on hard.

### Latency on the target laptop (200 clean sentences, batch size 1)

| system | threads | CER | sent acc | p50 ms | p95 ms | RSS MB |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B + LoRA, Q4_K_M | 8 | 0.0110 | 0.560 | 315.26 | 554.44 | 538.0 |
| … + lexicon draft (speculative, 4 tokens) | 8 | 0.0109 | 0.560 | 358.16 | 648.25 | – |
| character tagger | 1 | 0.0100 | 0.550 | 24.56 | 44.58 | 286.7 |
| character tagger | 4 | 0.0100 | 0.550 | 8.98 | 16.56 | 290.2 |
| character tagger, ONNX (onnxruntime) | 4 | 0.0100 | 0.550 | 16.70 | 26.78 | 318.3 |
| character tagger, ONNX (onnxruntime) | 1 | 0.0100 | 0.550 | 37.68 | 80.98 | 315.2 |

Files: `results/e4-laptop-*.json`, `results/lat-laptop-*.json`, `results/t2-laptop-tagger-*.json`,
`results/profile-laptop-q4_k_m-t8.profile.json`. Details in `docs/latency.md`.

**Large tagger** (`configs/tagger_large.yaml`, d 512 × 12 layers; sentence split: clean CER 0.0058 /
sentence accuracy 0.6857). Laptop on AC power, Windows Balanced plan, browser left open; three
interleaved rounds of 200 clean sentences, batch size 1 (`results/laptop-ac/`):

| system | threads | p95 ms per round | median p95 ms | median p50 ms |
|---|---|---|---|---|
| tagger (control) | 4 | 17.30 / 18.10 / 29.69 | 18.10 | 10.82 |
| large tagger | 1 | 110.82 / 112.47 / 116.55 | 112.47 | 63.11 |
| large tagger | 2 | 61.21 / 63.00 / 63.67 | 63.00 | 35.91 |
| large tagger | 4 | 41.79 / 64.83 / 70.07 | 64.83 | 29.27 |
| large tagger | 8 | 39.29 / 46.05 / 46.83 | 46.05 | 28.32 |
| large tagger | 16 | 28.92 / 31.63 / 31.66 | 31.63 | 19.69 |

The large tagger meets the 50 ms budget only with 8 or more threads, with little margin at 8. The
smaller tagger stays well inside it on 4 threads.

## Findings

1. **A fine-tuned 0.5B LLM restores diacritics well but cannot meet the budget on the target CPU.**
   LoRA lifts Qwen2.5-0.5B from unusable zero-shot (CER 0.5687) to CER 0.0129 / sentence accuracy
   0.5219, 3.3× the lexicon's sentence accuracy. On the laptop, the Q4_K_M model needs p95 554 ms,
   about 11× the budget, at the best thread count.
2. **The budget is structural, not a tuning problem.** On this CPU the prompt alone costs about
   80 ms and each generated token about 10 ms. Speculative decoding with the lexicon as the draft is
   accepted 3–5 tokens per pass but is still slower (p95 648–926 ms), and its outputs differ from
   greedy on 5–7 % of sentences because multi-token passes are not bit-identical.
3. **Quantisation is cheap in quality.** Q8_0 ≈ f16; Q4_K_M loses 1.3 (clean) to 2.8 (hard) points
   of sentence accuracy. It does not fix latency.
4. **A one-pass character tagger matches or beats the LLM and fits the budget.** The 14.4M-parameter
   tagger reaches clean CER 0.0080 / sentence accuracy 0.6022 and p95 16.56 ms on 4 CPU threads
   (44.58 ms on one), about 33× faster than the quantised LLM. It never rewrites letters, by
   construction, whereas the LLM rewrites about 3 % of sentences (`docs/error_analysis.md`).
5. **The LLM keeps a niche: typo repair.** A tagger can only add marks, so it cannot fix leftover
   Telex or typos (noisy CER 0.0358). Qwen2.5-1.5B trained on noisy pairs (E5) repairs them: noisy
   CER 0.0157 / sentence accuracy 0.5059, against 0.0358 / 0.2097 for the tagger and 0.0428 / 0.1919
   for the clean-trained 0.5B model, while staying close on clean input (0.0121 / 0.5675).
6. **LLMs are far more sample-efficient; the tagger wins on data, and data is free here.** At the
   LoRA run's budget (200k pairs, one epoch) the tagger is worse than the lexicon (CER 0.0698); it
   draws level with the LLM at about 1M pairs (5×) and passes it at 1.96M (about 10×). Training pairs for this task cost nothing
   — any accented Vietnamese text yields them by stripping marks — so the data requirement is not
   a practical obstacle.

## Caveats

- **Data confound.** The headline tagger saw ten times the training pairs of the LoRA runs (1.96M ×
  3 epochs vs 200k × 1); the data-efficiency table shows what equal data gives. The LLMs were not
  trained on more data, so whether they too keep improving with 1.96M pairs is open.
- E5 was trained with a bf16 base instead of the planned 4-bit (QLoRA) base, and was not run on the
  laptop.
- The latency runs use the first 200 clean test sentences; quality on those 200 differs slightly
  from the full test set.
- HF/PyTorch and llama.cpp give different outputs for the same merged model (clean2k CER 0.0121 vs
  0.0105); precision, merging and batch size were ruled out; the cause is open
  (`docs/quantization.md`).
- **Sentence-level split.** Every result above assigns sentences, not documents, to train and test,
  so near-duplicates and sentences of one document can span both. A rerun on a document-level
  split with a near-duplicate filter and an external test set is in progress (tags `ds-*`).
- The TCN was retrained on 300k pairs (its original checkpoint is lost), so its row is not an
  equal-data comparison with the 1.96M-pair taggers.
- Exact-match metrics count orthographic variants (`hoá` / `hóa`) as errors for every system.
- GPU greedy decoding is not bit-for-bit reproducible (a re-run moved sentence accuracy by 0.001).

## Recommendation for VIMEK

Use the character tagger for restoration while typing (4 threads, p95 16.56 ms on the target
laptop), with its top-5 beam for the candidate bar (`virestore.serve --model tagger`,
`POST /candidates`). For calling it from VIMEK's C++ without Python, `virestore.export_onnx` writes
an ONNX graph whose labels match PyTorch; under onnxruntime it is slower than PyTorch here (p95
26.78 ms on 4 threads, 52.24 ms on 2, 80.98 ms on 1), so it needs 4 threads to stay inside the budget. Keep the LLM off
the typing path; if typo repair is wanted, the noisy-trained 1.5B model can rework a sentence after
the user pauses, outside the per-keystroke budget.
