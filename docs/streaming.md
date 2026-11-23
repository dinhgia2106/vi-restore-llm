# Streaming: restoring diacritics while the sentence is still being typed

An input method does not receive finished sentences. This study runs each model on every word
prefix of each test sentence (`virestore.streaming`) and replays commit policies on those outputs,
so all policies are scored on the same predictions.

- `free`: every word on screen may still change.
- `wN`: only the last N+1 words may change; older words are frozen. `w0` never revises a word.

All runs use the document split. *clean* has 17 841 sentences, *external* (held-out shards) 14 000.
The TCN was run on the first 1 000 clean sentences because it restores one sentence per call.
Files: `results/stream-*.stream.json`.

## Results

Clean test set:

| model | word correct when first shown | sentence acc, `free` | revisions / 100 words, `free` | sentence acc, `w2` | revisions / 100 words, `w2` | sentence acc, `w0` |
|---|---|---|---|---|---|---|
| lexicon | 0.8494 | 0.1540 | 0.00 | 0.1540 | 0.00 | 0.1540 |
| TCN (1 000 sentences) | 0.6810 | 0.1120 | 52.36 | 0.1060 | 30.80 | 0.0120 |
| tagger | 0.8193 | 0.5811 | 20.82 | 0.5564 | 18.02 | 0.0487 |
| large tagger | 0.8226 | 0.6634 | 21.43 | 0.6190 | 18.53 | 0.0442 |
| tagger trained on prefixes | 0.8688 | 0.5484 | 14.16 | 0.5347 | 11.92 | 0.1173 |

External test set:

| model | word correct when first shown | sentence acc, `free` | revisions / 100 words, `free` | sentence acc, `w2` | revisions / 100 words, `w2` | sentence acc, `w0` |
|---|---|---|---|---|---|---|
| lexicon | 0.8497 | 0.1546 | 0.00 | 0.1546 | 0.00 | 0.1546 |
| tagger | 0.8231 | 0.5903 | 20.59 | 0.5666 | 17.88 | 0.0569 |
| large tagger | 0.8269 | 0.6691 | 21.10 | 0.6368 | 18.32 | 0.0559 |
| tagger trained on prefixes | 0.8720 | 0.5565 | 14.11 | 0.5472 | 11.81 | 0.1284 |

Large tagger by revision window (clean):

| policy | sentence acc | final word acc | revisions / 100 words | revisions that fix a word | revisions that break a word |
|---|---|---|---|---|---|
| `w0` | 0.0442 | 0.8226 | 0.00 | 0 | 0 |
| `w1` | 0.5172 | 0.9571 | 16.63 | 54 477 | 4 406 |
| `w2` | 0.6190 | 0.9664 | 18.53 | 59 342 | 5 780 |
| `w3` | 0.6350 | 0.9678 | 19.17 | 60 586 | 6 525 |
| `w5` | 0.6485 | 0.9690 | 19.87 | 61 907 | 7 376 |
| `free` | 0.6634 | 0.9705 | 21.43 | 64 543 | 9 479 |

## Latency per word

Target laptop on AC power, large and distilled taggers as int8 ONNX, 4 threads, one call per word
prefix over the first 200 clean sentences (4 663 calls), three rounds
(`results/laptop-ac-int8/stream-*-int8-t4-r*.stream.json`):

| model | median p50 ms | p95 ms per round | median p95 ms |
|---|---|---|---|
| large tagger, int8 | 8.61 | 18.84 / 18.17 / 18.52 | 18.52 |
| distilled tagger, int8 | 4.14 | 8.64 / 8.32 / 8.46 | 8.46 |

## Findings

1. **Freezing each word as soon as it appears destroys the result.** With `w0` the large tagger gets
   4.4 % of clean sentences right, against 66.3 % when words may change. A word's marks often depend
   on the words after it.
2. **A short revision window recovers most of the loss.** Letting the last three words change (`w2`)
   gives 0.6190 on clean, 93 % of the `free` score, and limits how far back the text can move.
3. **Most revisions help.** For the large tagger under `free`, 64 543 revisions fix a word and 9 479
   break a correct one.
4. **Training on prefixes is a trade-off.** Cutting half of the training sentences after a random word
   raises first-shown word accuracy by about 5 points and cuts revisions by about a third, but costs
   about 3 points of final sentence accuracy against the same-size tagger.
5. **The TCN is poorly suited to streaming at this training size.** Only 68.1 % of words are right
   when first shown, and it makes 52 revisions per 100 words.

## Caveats

- Prefixes end at word boundaries. Partial words typed between two spaces are not evaluated.
- Revisions are counted, but how distracting they are to a person was not measured.
- Test input is text with its marks removed, not real typing. No real typed test set exists yet.
- The quality rows use the PyTorch models. Int8 models were timed in streaming mode, and their
  full-sentence quality is in `docs/compression.md`.
