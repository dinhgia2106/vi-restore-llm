# Error analysis (E0, E2, E3)

Produced by `python -m virestore.analysis results/*.predictions.jsonl`; the per-run numbers are in
`results/*.analysis.json`. Test sets: clean and noisy test splits (19 956 sentences each), hard set
(2 801), E2 on the first 1 000 clean sentences.

**Base-letter fidelity** = share of outputs whose letters, with diacritics removed, equal the
input's (the model only added marks). **Wrong diacritics** / **rewritten words** are counted on
outputs with the reference word count: a wrong diacritic keeps the base letters, a rewrite does not.

| run | fidelity | more / fewer words | wrong diacritics | rewritten words |
|---|---|---|---|---|
| e0-lexicon-clean | 1.0000 | 0.0000 / 0.0000 | 61 452 | 0 |
| e2-qwen05-zeroshot-clean (1k) | 0.2700 | 0.2520 / 0.3880 | 6 217 | 387 |
| e3-qwen05-lora-clean | 0.9682 | 0.0024 / 0.0053 | 19 960 | 527 |
| e3-qwen05-lora-hard | 0.9650 | 0.0046 / 0.0036 | 3 444 | 82 |
| e3-qwen05-lora-noisy | 0.6293 | 0.0172 / 0.0236 | 25 146 | 21 095 |

## Findings

1. **Fine-tuning fixes the zero-shot failure mode.** Zero-shot, only 27 % of outputs keep the
   input's letters and 64 % change the word count (refusals, English answers, paraphrases). After
   LoRA, 96.8 % keep the letters and under 1 % change the word count.

2. **LoRA still rewrites about 3 % of sentences.** Examples from the clean set: `Duy → Đủ`,
   `tokbokki → tòktbokki`, `momen → mỏm`, `nhuộm → nhôm`; on the hard set `Ben → Benjamin`,
   `Chung → Chư`. These are rare (527 rewritten words vs 19 960 wrong diacritics on clean) but they
   are the errors a keyboard user notices most, because the word they typed changes. A cheap guard
   to test: when an output's base letters differ from the input, fall back to the lexicon for the
   affected words (or constrain decoding to the input's letters).

3. **The remaining errors are mostly genuine ambiguity.** The lexicon's top confusions are
   frequency-driven pairs (`cơ → có`, `cùng → cũng`, `bản → bạn`, `đề → để`, `nhưng ↔ những`); E3's
   are pairs that need wider context or world knowledge (`con → còn`, `qua → quá`, `cho → chờ`,
   `từ ↔ tự`) and proper nouns (`Thanh → Thành`).

4. **Part of the error count is orthographic, not wrong.** E3's most frequent "wrong diacritic" on
   clean is `hoá → hóa` (87): both are accepted spellings (old vs new tone-mark placement), and the
   corpus mixes them. The metrics compare exact strings, so every model is penalised for such
   variants; the effect is small (87 of 19 960) but a tone-placement normalisation before scoring
   would make the numbers fairer.

5. **Short sentences are hardest.** CER on 1–10-word sentences vs 31+ words: lexicon 0.0431 vs
   0.0328, E3 0.0231 vs 0.0112 (clean). Less context hurts both, the LLM more in relative terms. An
   IME sees short prefixes most of the time, so this is the regime that matters for VIMEK.

6. **Noisy input needs noisy training.** E3 was trained on clean pairs only. On noisy input it keeps
   many Telex leftovers as typed (`và → vaf`, `là → laf`, `một → mootj`, `được → đduowcj`) and
   fidelity drops to 0.63 (expected: noisy inputs need their letters changed). The lexicon does not
   repair noise at all — its rewritten-word count on noisy input (25 139) equals the identity
   baseline's. Training on `data/processed_noisy` is experiment E5.

## Latency (sandbox GPU, not the CPU budget)

E3 single-sentence p95 on the molab GPU is 487.97 ms (p50 290.5 ms), about ten times the 50 ms
budget before any CPU deployment. E4 (merge → GGUF Q8_0 / Q4_K_M, measured on the target laptop's
CPU) decides whether an LLM is viable inside the IME at all.
