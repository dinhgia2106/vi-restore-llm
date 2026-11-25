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

## Results

Pending: no boundary model has been trained yet.
