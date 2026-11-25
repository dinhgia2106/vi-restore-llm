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

## VIMEK prototype (Windows)

Branch `ai-candidates` of [VIMEK](https://github.com/dinhgia2106/VIMEK) runs suggestions next to
Telex, as an experimental option that is off by default (tray: Tùy chọn > Gợi ý dấu bằng AI).

- VIMEK keeps a copy of the text it has put on screen since the last break (Enter, arrows, a click,
  a focus change, `.` `!` `?`). It sends that text to `POST /candidates` on `127.0.0.1:8765` from a
  worker thread, so the keyboard hook never waits on the network for more than 100 ms.
- Only the **last three words** may change, as recommended in `docs/streaming.md`. They are sent as
  the user typed them, even if an earlier suggestion already changed them on screen, so a later word
  can still correct them. Earlier text is sent as shown.
- A window near the caret lists up to five distinct versions of those words. **Space** applies the
  first and types the space. **1-5** apply one and fix everything typed so far. **Esc** keeps the
  typed text, and a digit typed after it is a digit. A digit right after a space is always a digit,
  because the window closes when a word ends.
- Words the user marked with Telex are kept by the server (`lock_typed_words`). Without a server,
  VIMEK types Telex as before.

On the target laptop, an automated run (`VIMEK.exe --test-input`, tagged synthetic keys into a test
window) gave these results with the large int8 tagger:

| keys | text on screen |
|---|---|
| `hom nay troi dep qua ` | `hôm nay trời đẹp quá ` |
| `hom nay troi dep qua2` | `hôm nay trời đẹp qua` |
| `co`, Esc, `2 con` | `co2 con` |
| `co 2 con meo ` | `có 2 con mèo ` |
| `tieengs vieetj hay ` | `tiếng việt hay ` |
| `toi di hocc`, Backspace, ` ` | `tôi đi học ` |
| `Ha Noi mua thu ` | `Hà Nội mùa thu ` |

With the server stopped, the same keys gave the Telex text unchanged. This checks the mechanics
only. It is not a quality measurement, and it does not cover real applications yet (browsers,
Word, games), where caret position and text replacement can behave differently.
