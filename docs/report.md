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

## Headline results: document-level split

The first round of results (next section) split sentences, so one document could span train and
test. Everything was rerun on a stricter split: documents are assigned whole, test and dev sentences
that nearly duplicate a training sentence are removed (whole-sentence or 8-word-shingle match), and
an **external** test set is drawn from seven corpus shards never used for training. Sizes: 1 960 974
train, 16 553 dev, 17 841 test (2 470 removed), 2 514 hard, 14 000 external (3 590 removed). Files:
`results/ds-*.json`.

| system | clean CER | clean sent acc | hard CER | hard sent acc | noisy CER | noisy sent acc | external CER | external sent acc |
|---|---|---|---|---|---|---|---|---|
| lexicon | 0.0361 | 0.1540 | 0.0494 | 0.1193 | 0.0649 | 0.0733 | 0.0358 | 0.1546 |
| acausal TCN (300k pairs) | 0.0325 | 0.1282 | 0.0415 | 0.1153 | 0.0613 | 0.0586 | 0.0318 | 0.1277 |
| Qwen2.5-0.5B + LoRA | 0.0138 | 0.5119 | 0.0192 | 0.4586 | 0.0444 | 0.1920 | 0.0137 | 0.5136 |
| Qwen2.5-1.5B + LoRA, noisy training | 0.0137 | 0.5595 | 0.0177 | 0.5330 | 0.0173 | 0.4952 | 0.0131 | 0.5519 |
| character tagger | 0.0093 | 0.5811 | 0.0127 | 0.5410 | 0.0377 | 0.2031 | 0.0088 | 0.5903 |
| tagger trained on prefixes | 0.0104 | 0.5484 | 0.0141 | 0.5080 | 0.0384 | 0.1938 | 0.0097 | 0.5565 |
| distilled tagger | 0.0091 | 0.5900 | 0.0123 | 0.5509 | 0.0373 | 0.2058 | 0.0085 | 0.6019 |
| small tagger | 0.0130 | 0.4712 | 0.0177 | 0.4300 | 0.0407 | 0.1716 | 0.0124 | 0.4796 |
| small distilled tagger | 0.0129 | 0.4733 | 0.0175 | 0.4368 | 0.0407 | 0.1725 | 0.0122 | 0.4861 |
| large tagger | 0.0071 | 0.6634 | 0.0097 | 0.6229 | 0.0357 | 0.2301 | 0.0065 | 0.6691 |

The noisy rows of the lexicon and the 1.5B model use models trained on noisy pairs; the others were
trained on clean pairs. Against the sentence split, clean sentence accuracy fell by 2.1 points for
the tagger (0.6022 → 0.5811), 2.2 for the large tagger (0.6857 → 0.6634) and 1.0 for the 0.5B LoRA
model (0.5219 → 0.5119). The ranking did not change. By external domain the large tagger ranges from
0.5985 (encyclopedia) to 0.7425 (professional field).

**Best model: large tagger as int8 ONNX.** Clean CER 0.0071 / sentence accuracy 0.6625, external
0.0065 / 0.6691, 38.3 MB, median p95 22.48 ms on 4 threads on the target laptop
(`docs/compression.md`).

Follow-up studies on this split:

- `docs/streaming.md`: restoring while typing. With the last three words revisable, the large tagger
  keeps sentence accuracy 0.6190 of its 0.6634; freezing each word at once gives 0.0442.
- `docs/compression.md`: int8 costs at most 0.09 points of sentence accuracy and makes the ONNX graph
  2.4–3.1× faster. Distillation adds 0.2–1.2 points.
- `docs/calibration.md`: at a 99 % precision target the large tagger auto-applies 95.1 % of words on
  the external set (measured precision 99.05 %). A sentence-level gate covers only 5–11 %.

## Results on the sentence-level split

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

**Acausal TCN** (300k training pairs), same protocol with its own control runs
(`results/laptop-ac-tcn/`). On these 200 sentences the TCN scores CER 0.0289 / sentence accuracy
0.095, against 0.0100 / 0.550 for the tagger:

| system | threads | p95 ms per round | median p95 ms | median p50 ms |
|---|---|---|---|---|
| tagger (control) | 4 | 16.28 / 16.69 / 17.76 | 16.69 | 9.93 |
| TCN | 1 | 48.57 / 48.58 / 49.33 | 48.58 | 28.02 |
| TCN | 2 | 29.64 / 29.85 / 30.73 | 29.85 | 17.91 |
| TCN | 4 | 23.83 / 24.42 / 24.70 | 24.42 | 14.43 |
| TCN | 8 | 24.26 / 24.74 / 27.72 | 24.74 | 16.04 |
| TCN | 16 | 31.72 / 31.84 / 32.11 | 31.84 | 20.14 |

The TCN fits the budget at every thread count tried, but it is slower than the tagger on 4 threads
and far less accurate at its 300k-pair training size.

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
- **Sentence-level split.** The section "Results on the sentence-level split" and the numbered
  findings assign sentences, not documents, to train and test, so near-duplicates can span both.
  The headline section reruns every trained system on a document-level split.
- **No real typed test set.** All test input is corpus text with marks removed, plus synthetic noise.
  The external set comes from the same corpus family, and part of it is web crawl.
- Laptop latency of the TCN and of the PyTorch large tagger was measured on their sentence-split
  weights. The int8 latencies use the document-split weights.
- Latency on the laptop moves with background load: one scan of the int8 models was more than 2× slower
  at low thread counts (`results/laptop-ac-int8/`), and is kept alongside the five-round scan.
- Calibration thresholds were fitted on the PyTorch models and must be refitted for the int8 graph.
- The TCN was retrained on 300k pairs (its original checkpoint is lost), so its row is not an
  equal-data comparison with the 1.96M-pair taggers.
- Exact-match metrics count orthographic variants (`hoá` / `hóa`) as errors for every system.
- GPU greedy decoding is not bit-for-bit reproducible (a re-run moved sentence accuracy by 0.001).

## Recommendation for VIMEK

- **Model:** the large tagger as int8 ONNX, 4 threads (median p95 22.48 ms; per-word streaming p95
  18.52 ms). On weak machines use the distilled tagger as int8 (9.62 ms).
- **While typing:** run the model after each word and let only the last three words change.
- **Applying output:** auto-apply a word when its confidence clears a threshold fitted on dev for the
  int8 model; otherwise offer the top-5 candidates (`POST /candidates`).
- **Serving:** `virestore.serve --model tagger-onnx --onnx-file tagger.int8.onnx`, or call the graph
  from C++ with the contract in `docs/integration.md`. The graph takes one sentence per call.
- **LLM:** keep it off the typing path. If typo repair is wanted, the noisy-trained 1.5B model can
  rework a sentence after the user pauses (document split, noisy CER 0.0173 / sentence accuracy 0.4952
  against 0.0357 / 0.2301 for the large tagger).
