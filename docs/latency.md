# Latency on the target CPU

All numbers here were measured on the target laptop (Windows 11, Intel CPU with 32 logical
processors, llama-cpp-python 0.3.36 CPU build), with the Q4_K_M GGUF of the E3 model, 8 threads
(the fastest setting, see `docs/quantization.md`), one sentence at a time. Files: `results/profile-laptop-q4_k_m-t8.profile.json`
and `results/{e4-laptop,lat-laptop}-*.json`.

## Where the time goes

| measurement (median, 100 clean test sentences) | value |
|---|---|
| prompt tokens per call | 83 |
| generated tokens per sentence | 28.5 |
| full restoration | 316.04 ms |
| prompt + first generated token | 79.55 ms |

Cost of one forward pass over n new tokens at once:

| n | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms | 10.64 | 19.09 | 24.77 | 34.36 | 44.14 | 73.78 |

Reading the prompt alone already exceeds the 50 ms budget, and each generated token adds about
10 ms. **No decoding trick makes an autoregressive 0.5B model fit the budget on this CPU**: even a
draft that were always right would still pay the prompt (≈ 80 ms) plus one multi-token
verification pass.

## Speculative decoding with a lexicon draft

`GGUFRestorer` can take a `LexiconDraft` (`eval --draft lexicon --draft-tokens N`): the lexicon's
restoration of the rest of the sentence is proposed as draft tokens and llama.cpp verifies them in
one pass. Same 200 clean sentences, lexicon built from the first 300k training sentences:

| draft tokens | CER | sent acc | p50 ms | p95 ms | tokens per forward pass | identical to greedy |
|---|---|---|---|---|---|---|
| none (greedy) | 0.0110 | 0.560 | 315.26 | 554.44 | 1 | – |
| 4 | 0.0109 | 0.560 | 358.16 | 648.25 | 3.1 | 186 / 200 |
| 8 | 0.0106 | 0.565 | 380.80 | 775.54 | 4.0 | 188 / 200 |
| 16 | 0.0106 | 0.560 | 392.63 | 925.59 | 4.8 | 190 / 200 |

(tokens per forward pass = generated tokens / (draft rounds + one initial pass per sentence), from
`decode_stats` in the result files.)

- The draft is accepted often (3.1–4.8 tokens per pass instead of 1), yet every setting is **slower**,
  and longer drafts are slower still. From the forward-pass costs above, 3.1 tokens for the price
  of one 4-token pass (24.8 ms) should beat 3.1 single passes (33 ms); the measured slowdown means
  the overhead lies elsewhere. The likely cause, not measured separately, is llama-cpp-python's
  speculative path: it switches on `logits_all` (logits for every position, including the
  83-token prompt, over a 151k-token vocabulary) and samples each position in Python.
- Outputs are not bit-identical to greedy decoding (93–95 % identical): multi-token forward passes
  are not numerically identical to single-token ones, so near-ties flip. Quality is unchanged
  within noise.

## Conclusion for the IME

A full-sentence LLM restorer is about 11× over the budget on the target CPU, and speculative
decoding does not change that. Restoration inside the typing loop needs a model that labels the
input in one pass — see the character tagger (`virestore.tagger`). The LLM remains useful where
latency matters less: re-ranking the candidate bar when the user pauses, or as a teacher.
