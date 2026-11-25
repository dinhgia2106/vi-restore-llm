# Integrating the tagger into an input method

`virestore.serve --model tagger` (or `tagger-onnx`) applies the rules below through
`virestore.ime.ImeRestorer`. **A caller that runs `tagger.onnx` directly, such as VIMEK's C++ code,
must implement them itself.** The evaluation harness calls the raw tagger, so these rules do not
affect any number in `results/`.

## Rules the caller must apply

1. **Empty input.** Return empty or whitespace-only text without calling the model. A zero-length
   sequence makes the ONNX graph fail.
2. **Split before inference.** Send one sentence at a time. Cut anything longer than 192 characters
   at a clause or word boundary, and copy a single token longer than that unchanged. The model
   accepts up to 384 characters, but it was trained on single sentences of at most 40 words and
   degrades on longer input, so do not use 384 as the working limit.
3. **Strip marks first.** Marks in the input change the predictions for other words. Remove all
   diacritics before inference, then put back every word the user typed with marks.
4. **One tone per vowel run.** Mask invalid labels, then, if a run of adjacent vowels received two
   or more tones (`thỏả`), keep the assignment with the best total log-probability and give the
   other vowels their best toneless label. Words joined by punctuation (`và/hoặc`) are separate
   runs and are left alone.

## What the model cannot do

- **It does not repair typos or Telex.** It can only add marks to the letters it receives, so
  `dduwowcj` stays as is. The input method must resolve Telex before calling it.
- **The top candidate can be wrong when several forms are valid.** `hom nay troi dep qua` gives
  `hôm nay trời đẹp qua`, with `quá` as the second candidate. Both are real words, so show the
  candidate bar (`POST /candidates`) instead of committing the first output silently.

## Int8 model

The recommended graph is `tagger.int8.onnx`. Like every exported tagger it takes **one sentence per call** (`ids` has shape `[1, chars]`). Run one warm-up call at startup: the first call on the laptop took about 1.1 s, later ones a few milliseconds.

## VIMEK suggestions (Windows)

Shipped in VIMEK 0.1.0-alpha.10 as an experimental option, off by default. Since then, at the
user's request after real use, suggestions cover **only the word being typed** (from the sentence
start or the last space). Earlier words are never rewritten, and a word the user marks with Telex
gets no suggestion.

- VIMEK keeps a copy of the text it has put on screen since the last break (Enter, arrows, a click,
  a focus change, `.` `!` `?`). It sends that text to `POST /candidates` on `127.0.0.1:8765` from a
  worker thread. The keyboard hook waits at most 100 ms, and only on a digit.
- A window under the caret lists up to five forms of the current word. **1-5** apply one;
  **Space** is an ordinary space (since 2026-11-25, at the user's request it no longer applies the
  first candidate). **Esc** keeps the typed word, and a digit typed after it
  is a digit. A digit right after a space is always a digit.
- The caret is found on the worker thread: UI Automation first, then the Win32 caret, then MSAA,
  then the mouse position.
- **Model for this mode: `ds-tagger-prefix`**
  (`virestore.serve --model tagger --tagger runs/ds-tagger-prefix --threads 4`). With only the
  current word changeable, the model never sees the words after it. `docs/streaming.md` measures
  exactly this as "word correct when first shown": 0.8688 for the prefix-trained tagger, 0.8494 for
  the lexicon and 0.8226 for the large tagger on clean test text. So about one word in eight is
  still wrong when the first candidate is taken. The large tagger is the better model only when later words may
  revise earlier ones.

An automated run (`VIMEK.exe --test-input`, tagged synthetic keys into a Tk test window) with the
prefix tagger:

| keys | text on screen |
|---|---|
| `hom nay troi dep qua ` | `hôm nay trời đẹp quá ` |
| `hom nay troi dep qua2` | `hôm nay trời đẹp qua` |
| `co`, Esc, `2 con` | `co2 con` |
| `co 2 con meo ` | `có 2 con mèo ` |
| `tieengs vieetj hay ` | `tiếng việt hay ` |
| `toi di hocc`, Backspace, ` ` | `tôi đi học ` |
| `Ha Noi mua thu ` | `Hà Nội mua thu ` |
| `Tooi ddi hocj owr truwowngf ` | `Tôi đi học ở trường ` |
| `dduwowcj roi tot qua ` | `được rồi tốt quá ` |

`mua thu` shows the limit of current-word suggestions: `mùa` needs the next word. These runs check
the mechanics only. They are not a quality measurement, and caret placement in browsers and Office
has not been verified.
