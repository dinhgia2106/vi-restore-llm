---
license: mit
language:
  - vi
library_name: onnxruntime
pipeline_tag: token-classification
tags:
  - vietnamese
  - diacritics
  - input-method
  - onnx
  - int8
datasets:
  - opendatalab/WanJuan-Vietnamese
---

# Vietnamese diacritic tagger (int8 ONNX)

Restores Vietnamese tone marks and diacritics to text typed without them, fast enough to run inside a keyboard on a laptop CPU.

```
hom nay troi dep qua, chung ta di dao cong vien nhe
→ hôm nay trời đẹp quá, chúng ta đi dạo công viên nhé
```

This is the recommended model of the [vi-restore-llm](https://github.com/dinhgia2106/vi-restore-llm) study, which compared it with fine-tuned Qwen2.5 LLMs, an acausal TCN and a lexicon baseline under a 50 ms CPU latency budget.

## Model

- A 12-layer Transformer encoder over characters (d_model 512, 8 heads, 38.1M parameters), max 384 characters.
- It labels every vowel and `d` with one of 30 labels (modifier × tone) in **one forward pass**; other characters are copied. The output therefore always keeps the input's letters — it never rewrites words.
- Weights quantised to int8 with onnxruntime dynamic quantisation (`tagger.int8.onnx`, 40 MB).

## Results

Document-level split of WanJuan-Vietnamese (train and test share no document; near-duplicates of training sentences removed from test).

| Test set | Sentences | CER ↓ | WER ↓ | Sentence accuracy ↑ |
|---|---|---|---|---|
| Clean (held-out documents) | 17 841 | 0.0071 | 0.0296 | 0.6625 |
| External (7 held-out shards) | 14 000 | 0.0065 | 0.0275 | 0.6691 |

Latency: p95 **22.48 ms** per sentence on the target laptop (Intel, Windows 11), onnxruntime CPU, 4 threads, batch size 1 — median of five rounds of 200 sentences. For comparison, Qwen2.5-0.5B + LoRA as GGUF Q4_K_M needed p95 554 ms on the same machine.

## Use

```bash
pip install numpy onnxruntime
python restore.py "toi dang hoc tri tue nhan tao o Da Nang"
# Tôi đang học trí tuệ nhân tạo ở Đà Nẵng
```

```python
from restore import DiacriticTagger
tagger = DiacriticTagger(".", threads=4)
print(tagger.restore("hom nay troi dep qua"))
```

The graph takes one sentence per call: input `ids` (int64, `[1, chars]`; 0 = padding, 1 = unknown, others are indices into `vocab` in `config.json` offset by 2), output `logits` (`[1, chars, 30]`). Label = modifier × 6 + tone, with modifiers (none, breve, circumflex, horn, stroke) and tones (none, grave, acute, hook, tilde, dot below). Run each sentence at its own length: int8 dynamic quantisation makes padded positions change the result.

## Limitations

- Input must be the same letters as the intended text. It cannot fix typos or leftover Telex keystrokes (`tieengs`) — that needs a sequence-to-sequence model; see the noisy-trained Qwen2.5-1.5B in the study.
- Trained and tested on web text with diacritics removed, not on text people actually typed.
- Exact-match metrics count orthographic variants (`hoá` / `hóa`) as errors.

## Data and license

Training data: [WanJuan-Vietnamese](https://huggingface.co/datasets/opendatalab/WanJuan-Vietnamese) (OpenDataLab, CC BY 4.0; Yu et al., 2025, arXiv:2501.14506), filtered and converted to training pairs as described in the repository's `data/README.md`. Model and code: MIT.
