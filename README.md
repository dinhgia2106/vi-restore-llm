# vi-restore-llm

**Can a small LLM beat a purpose-built model at restoring Vietnamese diacritics — fast enough to run inside a keyboard?**

People type Vietnamese without tone marks all the time (`hom nay troi dep qua`), and an input method has to put them back (`hôm nay trời đẹp quá`) while the user is still typing. This repo fine-tunes small open LLMs (0.5–2B) for that job, benchmarks them against a task-specific acausal TCN, quantises the best one, and serves it locally for the [VIMEK](https://github.com/dinhgia2106/VIMEK) input method.

```
corpus ─► pairs (strip marks + Telex/typo noise) ─► LoRA / QLoRA fine-tune ─► eval harness ─► merge ─► GGUF int4 ─► local server ─► VIMEK
```

## Research question

> Under a **50 ms p95 CPU latency budget**, which gives the best restoration quality: a frequency lexicon, a character-level TCN, or a LoRA-tuned small LLM — and how much quality does int4 quantisation cost?

Hypotheses and the full experiment matrix are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Results

Filled in by `python -m virestore.eval`; see [`results/summary.md`](results/summary.md).

| Model | Params | CER ↓ | WER ↓ | Sentence acc ↑ | p95 latency (CPU) | RAM |
|---|---|---|---|---|---|---|
| Identity (floor) | – | | | | | |
| Lexicon (unigram + bigram) | – | | | | | |
| Acausal TCN ([repo](https://github.com/dinhgia2106/Vietnamese-diacritic-restoration)) | | | | | | |
| Qwen2.5-0.5B, zero-shot | 0.5B | | | | | |
| Qwen2.5-0.5B + LoRA | 0.5B | | | | | |
| Qwen2.5-0.5B + LoRA, GGUF Q4_K_M | 0.5B | | | | | |
| Qwen2.5-1.5B + QLoRA (noisy input) | 1.5B | | | | | |

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"          # core + tests
uv pip install -e ".[train,serve,gguf]"         # fine-tuning, serving, llama.cpp

# 1. data — WanJuan-Vietnamese shards (see data/README.md), or any Vietnamese .txt / .jsonl
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed --limit 2000000
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed_noisy --noisy --limit 2000000
python -m virestore.hardset --train data/processed/train.jsonl --test data/processed/test.jsonl --out data/processed/test_hard.jsonl

# 2. baselines
python -m virestore.eval --model identity --test data/processed/test.jsonl
python -m virestore.eval --model lexicon --train data/processed/train.jsonl --test data/processed/test.jsonl
python -m virestore.eval --model tcn --tcn-repo ../Vietnamese-diacritic-restoration --test data/processed/test.jsonl

# 3. LLM: zero-shot, then fine-tuned
python -m virestore.eval --model hf --hf-model Qwen/Qwen2.5-0.5B-Instruct --test data/processed/test.jsonl --limit 1000
python -m virestore.train_lora --config configs/qwen05_lora.yaml
python -m virestore.eval --model hf --hf-model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05-lora --test data/processed/test.jsonl

# 4. on-device
python -m virestore.merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/qwen05-lora --out runs/qwen05-lora-merged
#    convert + quantise with llama.cpp → docs/quantization.md
python -m virestore.eval --model gguf --gguf models/qwen05-lora-q4_k_m.gguf --test data/processed/test.jsonl
python -m virestore.serve --model gguf --gguf models/qwen05-lora-q4_k_m.gguf   # POST /restore
```

## Data

Sentences come from **[WanJuan-Vietnamese](https://opendatalab.com/OpenDataLab/WanJuan-Vietnamese)**, the Vietnamese subset of OpenDataLab's WanJuan3.0 web corpus ([Yu et al., 2025](https://arxiv.org/abs/2501.14506), CC BY 4.0). The corpus itself is not redistributed; [data/README.md](data/README.md) lists the filtering and changes applied and the citation.

## Tasks

| Task | Input | Target |
|---|---|---|
| **clean** | diacritics stripped: `Hom nay troi dep qua` | `Hôm nay trời đẹp quá` |
| **noisy** | plus leftover Telex and typos: `Buoi sang o Da nawngx` | `Buổi sáng ở Đà Nẵng` |

Splits are assigned by a hash of the sentence, so the same sentence can never appear in both train and test.

## Metrics

- **CER / WER** — edit distance over characters / words (robust to the length changes noisy input causes)
- **Word accuracy** and **sentence exact match**
- **Latency** — single-sentence p50 / p95, measured the way an IME calls the model (batch size 1, after warm-up)
- **Memory** — resident set size of the process

## Layout

```
src/virestore/
  text.py         normalisation, diacritic stripping, sentence splitting
  noise.py        Telex / typo noise generator
  data.py         corpus → train/dev/test JSONL
  hardset.py      test subset of sentences with highly ambiguous syllables
  metrics.py      CER, WER, word and sentence accuracy
  prompts.py      one prompt format for training and inference
  models/         identity, lexicon, TCN adapter, Hugging Face LLM, GGUF (llama.cpp)
  eval.py         quality + latency + memory harness → results/
  train_lora.py   LoRA / QLoRA SFT (loss on the answer only)
  merge.py        merge adapter → full model for conversion
  serve.py        FastAPI server for VIMEK
configs/          experiment configs
docs/             experiment plan, quantisation guide
data/sample/      tiny hand-written corpus for tests
tests/            unit tests (pytest)
```
