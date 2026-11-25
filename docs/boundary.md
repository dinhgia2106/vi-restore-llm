# Word boundaries: input typed without spaces

The diacritic models in this repository need the spaces between syllables. An input method in the
style of Chinese pinyin IMEs lets the user type a whole phrase without spaces or marks
(`homnaytroidepqua`) and pick a reading. That needs word boundaries as well as marks.

## Plan

- `virestore.boundary` trains the tagger's character Transformer with a two-class head on
  sentences with marks and spaces removed. Each character is labelled 1 if a space follows it. The
  last character is not scored, since whether a space follows the last typed character is
  unknown. 30 % of training sentences are cut at a random character, as if the user were still
  typing.
- The boundaries feed the existing diacritic tagger. `virestore.eval --model segment-tagger`
  removes the spaces from each test input, segments it, tags it and scores the result against the
  usual references. Its rows are therefore comparable with the tagger rows that receive the true
  spaces, and the difference is the cost of finding the boundaries.
- Training and evaluation use the document split (`configs/boundary_ds.yaml`, notebook cell
  `boundary`, jobs `bd_data` and `bd_gpu`, tags `bd-boundary-*` and `ds-seg-*`).

## Reference: NextKey student on the target laptop

The user's team project NextKey_2026 already predicts boundaries, marks and capitalisation jointly.
Its student (2 078 648 parameters, checkpoint `checkpoint_step_359851.pt` of run
`gch-stu-006-graph-selective-kd-seed42-r1`, SHA-256 `28580c4f…`) was timed here as a reference
before deciding to train a boundary model from scratch.

Protocol: the target laptop on AC power, NextKey's own code and decoder (top-3), batch size 1, the
first 200 sentences of `data/processed/test.jsonl` converted to NextKey's unspaced lowercase input
(mean 73.7 characters), five warm-up calls, five interleaved rounds per setting, time from the raw
string to the top-3 sentences. INT8 is NextKey's `torch_ao_dynamic_int8` configuration
(`configs/quantization/int8.yaml`), built on the laptop. Files: `results/laptop-nextkey/`.

| model | threads | median p95 ms | p95 per round | median p50 ms |
|---|---|---|---|---|
| FP32 | 1 | 17.51 | 15.42 / 16.94 / 17.51 / 18.10 / 21.13 | 10.18 |
| FP32 | 2 | 17.34 | 14.96 / 15.77 / 17.34 / 18.63 / 27.71 | 9.48 |
| FP32 | 4 | 21.84 | 17.25 / 18.78 / 21.84 / 26.15 / 28.57 | 12.79 |
| FP32 | 8 | 26.68 | 18.54 / 23.46 / 26.68 / 28.66 / 33.62 | 12.65 |
| INT8 | 1 | 36.91 | 35.69 / 35.85 / 36.91 / 39.54 / 51.35 | 24.60 |
| INT8 | 2 | 44.41 | 38.40 / 38.91 / 44.41 / 50.08 / 50.68 | 27.27 |
| INT8 | 4 | 55.55 | 42.40 / 51.00 / 55.55 / 57.90 / 60.21 | 29.71 |
| INT8 | 8 | 62.70 | 46.85 / 61.85 / 62.70 / 64.90 / 77.81 | 36.95 |

- On this laptop the dynamic INT8 model is about twice as slow as FP32, and more threads make both
  slower. PyTorch on Windows offers only the `onednn` quantized engine, which is the likely cause.
- The model forward pass is almost all of the time. Decoding the top-3 took under 2.1 ms at p95.
- The rounds spread widely (FP32 at 2 threads: 14.96 to 27.71 ms), and the machine was in use, so
  these are indicative.
- `results/laptop-nextkey/samples.json` holds a few outputs as a sanity check. They are not a
  quality measurement: NextKey was trained on WanJuan-Vietnamese, which these sentences come from.

## Results

Pending: no boundary model has been trained yet.
