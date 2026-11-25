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

## What the errors are

The first boundary model already misplaces only 0.33 % of the boundaries between two letters on
clean test text (`bd-boundary-clean`, `error_rate_letters`), against 1.6 % next to digits or
punctuation. Most sentences it gets wrong contain one of two kinds of error (examples from
`results/bd-boundary-clean.boundary.json`):

- **Strings that are valid Vietnamese both ways.** Without marks and spaces the input does not
  decide: `denhanh` is `de nhanh` (đẻ nhánh) or `den hanh`, `chathap` is `chat hap` or `cha that`,
  `quanho` is `qua nho` or `quan ho`, `trenhat` is `tren hat` or `tre nhat`. Only context, or the
  user, can choose; this is why the input method shows several candidates.
- **Noise in the references**, such as `tudo`, `manh meden` or `tac giaDao` in the corpus text,
  and inconsistent spacing around hyphens and brackets.

A first-candidate exact rate of 98 % on whole sentences is therefore not a realistic target. The
tables below measure what an input method needs: short chunks, first candidate and top 3.

## Status

Best for an input method so far: the **large model trained on mid-sentence chunks, with the lexicon
at −4** (`runs/ds-boundary-large-span` + `lexicon.json`, in the private Hugging Face repo
`dinhgia2106/vi-restore-llm-docsplit` under `ds-boundary-large-span/`). On clean test chunks its
first candidate is exact for 99.4 % (1 word), 98.6 % (2), 98.2 % (3), 97.8 % (4) and 96.1 % (8
words); the right segmentation is among the first three for at least 98.6 % at every length up to
eight words. Served with the int8 prefix tagger it answers in a median p95 of 29.79 ms at 4
threads on the target laptop (section at the end).

## Results

All rows are on the document split. *clean*: 17 841 test sentences; *external*: 14 000 sentences
from seven held-out shards; *hard*: 2 514 clean sentences full of ambiguous syllables. The lexicon
penalty for unknown tokens was chosen on dev (table at the end); test was scored only at that value
(−4), plus −8, which had been scheduled before the dev sweep finished.

### Short chunks (what an input method sees)

Each clean test sentence is cut into consecutive chunks of n words made only of letters, and every
chunk is segmented on its own. Exact segmentation of the first candidate (top 1) and of any of the
first three (top 3):

| model | decoding | 1 word | 2 words | 3 words | 4 words | 6 words | 8 words |
|---|---|---|---|---|---|---|---|
| small (4.9M), sentences + prefixes | model only | 97.9 / 99.9 | 97.2 / 99.2 | 97.0 / 98.8 | 96.5 / 98.5 | 95.3 / 97.7 | 94.3 / 97.1 |
| small (4.9M), sentences + prefixes | lexicon −4 | 97.9 / 99.9 | 97.4 / 99.6 | 97.2 / 99.4 | 96.9 / 99.2 | 95.8 / 98.7 | 94.8 / 98.3 |
| large (14M), sentences + prefixes | model only | 97.7 / 99.9 | 97.4 / 99.2 | 97.4 / 99.0 | 97.0 / 98.8 | 96.2 / 98.2 | 95.5 / 97.7 |
| large (14M), sentences + prefixes | lexicon −4 | 97.7 / 99.9 | 97.6 / 99.6 | 97.6 / 99.5 | 97.2 / 99.3 | 96.5 / 98.9 | 95.8 / 98.6 |
| small, + mid-sentence chunks | model only | 99.4 / 99.9 | 98.3 / 99.3 | 97.7 / 98.9 | 97.0 / 98.5 | 95.7 / 97.6 | 94.4 / 96.8 |
| small, + mid-sentence chunks | lexicon −4 | 99.4 / 99.9 | 98.5 / 99.7 | 98.0 / 99.5 | 97.3 / 99.2 | 96.1 / 98.7 | 95.0 / 98.2 |
| large, + mid-sentence chunks | model only | 99.5 / 99.9 | 98.5 / 99.4 | 98.1 / 99.1 | 97.6 / 98.8 | 96.6 / 98.1 | 95.8 / 97.6 |
| large, + mid-sentence chunks | lexicon −4 | 99.4 / 99.9 | 98.6 / 99.7 | 98.2 / 99.5 | 97.8 / 99.4 | 96.9 / 99.0 | 96.1 / 98.6 |

Cells are top 1 / top 3, in %.

### Whole sentences

| model | decoding | clean F1 | clean exact | clean exact, letters only | external F1 | external exact | hard exact |
|---|---|---|---|---|---|---|---|
| small (4.9M), sentences + prefixes | model only | 0.9926 | 83.5 | 87.2 | 0.9925 | 82.4 | 87.4 |
| small (4.9M), sentences + prefixes | lexicon −4 | 0.9929 | 84.5 | 88.4 | 0.9926 | 83.4 | 88.1 |
| small (4.9M), sentences + prefixes | lexicon −8 | 0.9920 | 82.8 | 86.5 | 0.9912 | 81.1 | 86.8 |
| large (14M), sentences + prefixes | model only | 0.9944 | 86.9 | 90.6 | 0.9941 | 85.9 | 89.5 |
| large (14M), sentences + prefixes | lexicon −4 | 0.9945 | 87.5 | 91.3 | 0.9941 | 86.6 | 89.9 |
| large (14M), sentences + prefixes | lexicon −8 | 0.9937 | 85.9 | 89.5 | 0.9929 | 84.3 | 88.9 |
| small, + mid-sentence chunks | model only | 0.9917 | 81.6 | 85.4 | 0.9917 | 80.8 | 86.5 |
| small, + mid-sentence chunks | lexicon −4 | 0.9921 | 83.0 | 87.0 | 0.9920 | 82.2 | 87.5 |
| large, + mid-sentence chunks | model only | 0.9936 | 85.3 | 89.1 | 0.9935 | 84.5 | 88.5 |
| large, + mid-sentence chunks | lexicon −4 | 0.9937 | 86.1 | 90.0 | 0.9935 | 85.2 | 89.0 |

*exact*: every boundary of the sentence right, in %. *letters only*: ignoring boundaries next to
digits or punctuation, which an input method does not have to find (it splits typed text there).

### Boundaries and marks together

`virestore.eval --model segment-tagger`: spaces removed from the input, small boundary model
(sentences + prefixes, no lexicon), then the diacritic tagger; scored against the usual
references (files `results/ds-seg-*.json`).

| tagger | test set | CER | sentence acc | EM@3 | EM@5 |
|---|---|---|---|---|---|
| tagger-large | clean | 0.0121 | 0.5700 | 0.6680 | 0.6998 |
| tagger-large | hard | 0.0142 | 0.5577 | 0.6591 | 0.6977 |
| tagger-large | external | 0.0117 | 0.5713 | 0.6636 | 0.6914 |
| tagger-prefix | clean | 0.0153 | 0.4780 | 0.5887 | 0.6236 |
| tagger-prefix | hard | 0.0186 | 0.4598 | 0.5648 | 0.6094 |
| tagger-prefix | external | 0.0149 | 0.4784 | 0.5889 | 0.6211 |

With the true spaces the large tagger reaches sentence accuracy 0.6634 (clean), the prefix tagger
0.5484 (`docs/report.md`).

### Lexicon penalty, chosen on dev

| model | unknown-token penalty | sentence exact | 2 words top 1 | 4 words top 1 | 8 words top 1 |
|---|---|---|---|---|---|
| small | model only | 83.8 | 97.4 | 96.6 | 94.5 |
| small | −4 | 84.8 | 97.6 | 97.0 | 95.2 |
| small | −6 | 84.1 | 97.5 | 96.9 | 94.9 |
| small | −8 | 83.4 | 97.4 | 96.7 | 94.7 |
| small | −12 | 82.4 | 97.4 | 96.6 | 94.4 |
| large | model only | 87.0 | 97.6 | 97.2 | 95.8 |
| large | −4 | 87.3 | 97.7 | 97.4 | 96.0 |
| large | −8 | 85.8 | 97.6 | 97.1 | 95.5 |

## Serving: `POST /chunk` and laptop latency

`virestore.serve --boundary DIR` adds `POST /chunk` (`ChunkSuggester`): the input method sends the
context before the chunk, the chunk as shown and the keys pressed. The server segments the keys
(the Telex engine can show `anhsang` as `ánhang`), keeps a single syllable the user marked with
Telex as typed, tags every segmentation in one pass with the last six context words, and returns up
to five candidates: three readings of the best segmentation, then other segmentations made only of
common tokens, then more readings.

Latency of one suggestion on the target laptop on AC power: 200 chunks of one to four words from
`data/processed/test.jsonl` (mean 8.2 characters), each with its sentence context, five rounds,
int8 prefix tagger (`ds-tagger-prefix/tagger.int8.onnx`, exported on the laptop), lexicon −4. Files:
`results/laptop-chunk/`.

| boundary model | threads | median p95 ms | p95 per round | median p50 ms |
|---|---|---|---|---|
| large, chunks | 4 | 29.79 | 26.40 / 28.40 / 29.79 / 30.49 / 30.70 | 15.35 |
| large, chunks | 2 | 45.40 | 39.06 / 40.05 / 45.40 / 47.00 / 56.22 | 22.61 |
| small, chunks | 4 | 26.97 | 26.04 / 26.15 / 26.97 / 27.31 / 30.69 | 12.39 |
| small, chunks | 2 | 33.97 | 32.97 / 33.91 / 33.97 / 34.85 / 34.87 | 15.04 |

The boundary model takes about 3 ms; the rest is the tagger. A first version that ran the PyTorch
tagger three times per suggestion over the whole sentence context had a p95 of about 130 ms at
4 threads and was replaced. The quality of the int8 prefix tagger and of the full deployed pipeline
on the test sets is being measured (job `prefix_int8b`).
