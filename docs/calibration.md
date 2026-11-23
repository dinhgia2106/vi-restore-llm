# Calibration: when can an input method apply a restoration without asking?

An IME can either apply the model's output directly or show candidates. That decision needs a
confidence score that can be trusted. `virestore.calibration` scores a word by the product of the
top-label probabilities of its letters, and a sentence by the product over all its letters.

Temperature and thresholds are fitted on the **dev** set only. All numbers below are measured on the
**external** test set: 14 000 sentences from held-out shards, 289 285 aligned words. Files:
`results/calib-*-external.calibration.json`.

## Calibration and ranking

| model | level | accuracy | ECE, raw | ECE, temperature scaled | AUROC, raw | temperature |
|---|---|---|---|---|---|---|
| tagger | word | 0.9629 | 0.0057 | 0.0021 | 0.9502 | 1.15 |
| tagger | sentence | 0.5903 | 0.0102 | 0.0584 | 0.8780 | 1.15 |
| distilled tagger | word | 0.9640 | 0.0145 | 0.0020 | 0.9486 | 1.40 |
| distilled tagger | sentence | 0.6019 | 0.0918 | 0.0456 | 0.8714 | 1.40 |
| large tagger | word | 0.9726 | 0.0060 | 0.0016 | 0.9548 | 1.20 |
| large tagger | sentence | 0.6691 | 0.0249 | 0.0512 | 0.8790 | 1.20 |

## Auto-apply at a target precision

For each target, the lowest threshold reaching it on dev is applied to test. *Coverage* is the share
of words or sentences above the threshold; *precision* is how many of those are right. Raw
(uncalibrated) confidence:

**Words**

| model | target | dev threshold | test coverage | test precision |
|---|---|---|---|---|
| tagger | 0.98 | 0.6957 | 0.9610 | 0.9810 |
| tagger | 0.99 | 0.8996 | 0.9159 | 0.9907 |
| distilled tagger | 0.98 | 0.7678 | 0.9636 | 0.9809 |
| distilled tagger | 0.99 | 0.9537 | 0.9229 | 0.9904 |
| large tagger | 0.98 | 0.5887 | 0.9845 | 0.9808 |
| large tagger | 0.99 | 0.8612 | 0.9511 | 0.9905 |

**Sentences**

| model | target | dev threshold | test coverage | test precision |
|---|---|---|---|---|
| tagger | 0.90 | 0.7747 | 0.3916 | 0.9210 |
| tagger | 0.95 | 0.9206 | 0.2458 | 0.9590 |
| tagger | 0.99 | 0.9897 | 0.0499 | 0.9928 |
| distilled tagger | 0.90 | 0.8960 | 0.4121 | 0.9149 |
| distilled tagger | 0.95 | 0.9786 | 0.2455 | 0.9633 |
| distilled tagger | 0.99 | 0.9962 | 0.0693 | 0.9866 |
| large tagger | 0.90 | 0.7518 | 0.5379 | 0.9116 |
| large tagger | 0.95 | 0.9218 | 0.3616 | 0.9597 |
| large tagger | 0.99 | 0.9883 | 0.1131 | 0.9880 |

## Findings

1. **Word-level confidence is reliable enough for auto-apply.** At a 99 % target, the large tagger
   applies 95.1 % of words at a measured precision of 99.05 % on shards it never saw.
2. **Decide per word, not per sentence.** A sentence-level gate at about 99 % precision covers only
   5–11 % of sentences.
3. **Temperature scaling helps words, not sentences.** It lowers word-level ECE for all three models.
   At sentence level it makes ECE worse for the tagger and the large tagger: a temperature fitted per
   character does not account for errors being correlated within a sentence.
4. **Dev thresholds transfer, with a small miss at the strictest sentence targets.** The distilled and
   large taggers reach 98.66 % and 98.80 % against a 99 % sentence target.

## Caveats

- Thresholds were fitted on the PyTorch models. **Refit them for the int8 ONNX model before using
  them**, since its probabilities can differ slightly.
- Words count only in sentences whose output and reference have the same number of words.
- The study uses clean input. Thresholds on noisy or real typed input are unmeasured.
- Precision here means exact match with the corpus text, so orthographic variants (`hoá` / `hóa`)
  count as errors.
