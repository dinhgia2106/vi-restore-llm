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

Filled in by `python -m virestore.eval`; every run, including the noisy and hard test sets, is in [`results/summary.md`](results/summary.md). The table shows the **clean** test split (19 956 sentences, 2M-sentence build from WanJuan-Vietnamese).

| Model | Params | CER ↓ | WER ↓ | Sentence acc ↑ | p95 latency (CPU) | RAM |
|---|---|---|---|---|---|---|
| Identity (floor) | – | 0.2111 | 0.8284 | 0.0000 | | |
| Lexicon (unigram + bigram) | – | 0.0342 | 0.1426 | 0.1576 | | |
| Acausal TCN ([repo](https://github.com/dinhgia2106/Vietnamese-diacritic-restoration)) | | | | | | |
| Qwen2.5-0.5B, zero-shot¹ | 0.5B | 0.5687 | 1.0144 | 0.0000 | | |
| Qwen2.5-0.5B + LoRA² | 0.5B | 0.0129 | 0.0493 | 0.5219 | | |
| Qwen2.5-0.5B + LoRA, GGUF Q4_K_M³ | 0.5B | 0.0108 | 0.0423 | 0.5780 | 554.44 ms⁴ | 538 MB⁴ |
| Qwen2.5-1.5B + QLoRA (noisy input) | 1.5B | | | | | |
| **Character tagger** (`virestore.tagger`)⁵ | 14.4M | **0.0080** | **0.0333** | **0.6022** | **16.56 ms**⁵ | 290 MB⁵ |

¹ First 1 000 sentences of the clean test split. Without fine-tuning the model often refuses, answers in English, rewrites the sentence or returns the input unchanged, so it scores below the identity floor.

² `configs/qwen05_lora.yaml`: LoRA r = 16, 200k clean training pairs, one epoch. On the hard set it reaches CER 0.0181 / sentence acc 0.4681 (lexicon: 0.0459 / 0.1317); on noisy input, which it was not trained on, CER 0.0428 / sentence acc 0.1919 (lexicon trained on noisy pairs: 0.0630 / 0.0724).

³ First 2 000 sentences of the clean test split, llama.cpp on CPU. On the same sentences the f16 GGUF of the same merged model scores 0.0105 / 0.0406 / 0.5910 and Q8_0 0.0105 / 0.0411 / 0.5895, so Q4_K_M costs 0.0003 CER and 1.3 points of sentence accuracy (hard set: 2.8 points). The HF/PyTorch run of the same adapter scores lower on these sentences (0.0121 / 0.0472 / 0.5440); the cause is not yet known — see [docs/quantization.md](docs/quantization.md).

⁴ Target laptop (Intel, 32 logical CPUs, Windows 11), llama-cpp-python 0.3.36 on CPU, 8 threads, batch size 1, 200 clean sentences (`results/e4-laptop-q4_k_m-t8.json`); RAM is the process RSS. That is **about 11× the 50 ms budget**. 8 threads was the fastest setting tried (p95: 4 threads 623.84 ms, 16 threads 586.44 ms, 32 threads 797.51 ms); Q8_0 at 8 threads: 665.47 ms.

⁵ `configs/tagger.yaml`: a Transformer encoder over characters that labels every vowel (and `d`) with one of 30 diacritic labels in a single pass, so it cannot rewrite letters. Trained on all 1.96M clean training pairs for 3 epochs — **ten times the data of the LoRA run**, so this row does not isolate the architecture. Hard set: CER 0.0114 / sentence acc 0.5587; noisy input (not trained on): 0.0358 / 0.2097. Latency: target laptop, PyTorch 2.15 CPU, 4 threads, batch size 1, 200 clean sentences (`results/t2-laptop-tagger-t4.json`; on those sentences CER 0.0100 / sentence acc 0.550 against 0.0110 / 0.560 for Q4_K_M). 1 thread: 44.58 ms, 2 threads: 25.16 ms, 8 threads: 16.28 ms.

**Candidate bar (EM@k).** Share of sentences whose reference is among the top k candidates (`eval --nbest 5`, runs tagged `-nbest`; full test sets):

| Model | test set | EM@1 | EM@3 | EM@5 |
|---|---|---|---|---|
| Lexicon | clean | 0.1824 | 0.2970 | 0.3513 |
| Qwen2.5-0.5B + LoRA | clean | 0.5981 | 0.7397 | 0.7757 |
| Character tagger | clean | 0.6022 | 0.7324 | 0.7719 |
| Lexicon | hard | 0.1653 | 0.2699 | 0.3210 |
| Qwen2.5-0.5B + LoRA | hard | 0.5491 | 0.6951 | 0.7351 |
| Character tagger | hard | 0.5587 | 0.6944 | 0.7283 |
| Lexicon | noisy | 0.0805 | 0.1272 | 0.1481 |
| Qwen2.5-0.5B + LoRA | noisy | 0.2173 | 0.2753 | 0.2922 |
| Character tagger | noisy | 0.2097 | 0.2527 | 0.2644 |

Beam search alone lifts the LoRA model's top-1 above its greedy output (clean 0.5981 vs 0.5209 in the same run). Greedy outputs on the GPU are not bit-for-bit reproducible: the `-nbest` re-run of the clean set scored sentence accuracy 0.5209 against 0.5219 in the original E3 run; the table above keeps the original.

What the errors look like (rewritten words, ambiguous pairs, sentence length, noisy input): [docs/error_analysis.md](docs/error_analysis.md).

Latency and RAM columns are filled only from runs on the target laptop (CPU, batch size 1). The other runs were made on a molab sandbox (virtualised CPU; the LLM on a GPU), so their latency and memory, recorded in `results/`, are not comparable to the 50 ms budget.

## Use the model

The model that fits the IME budget is the character tagger (findings: [docs/report.md](docs/report.md)). It is kept in a private Hugging Face repo for now.

```bash
uv pip install -e ".[serve]" torch            # a CPU build of torch is enough
hf download dinhgia2106/vi-restore-llm-tagger --local-dir runs/tagger   # needs access to the private repo
python -m virestore.serve --model tagger --tagger runs/tagger --threads 4 --port 8765
```

```bash
curl -s localhost:8765/restore    -d '{"text": "hom nay troi dep qua"}'          -H 'content-type: application/json'
curl -s localhost:8765/candidates -d '{"text": "toi di hoc", "k": 5}'           -H 'content-type: application/json'
```

`/restore` returns one sentence, `/candidates` the top-k for a candidate bar. The tagger only restores diacritics: it never changes letters, so it does not fix typos or leftover Telex.

## Quick start

The GPU experiments run in a marimo notebook on molab: [notebooks/experiments.py](notebooks/experiments.py), guide in [docs/molab.md](docs/molab.md). Locally:

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

Sentences come from **[WanJuan-Vietnamese](https://huggingface.co/datasets/opendatalab/WanJuan-Vietnamese)**, the Vietnamese subset of OpenDataLab's WanJuan3.0 web corpus ([Yu et al., 2025](https://arxiv.org/abs/2501.14506), CC BY 4.0). The corpus itself is not redistributed; [data/README.md](data/README.md) lists the shards used (one per domain), the filtering and changes applied, and the citation.

## Tasks

| Task | Input | Target |
|---|---|---|
| **clean** | diacritics stripped: `Hom nay troi dep qua` | `Hôm nay trời đẹp quá` |
| **noisy** | plus leftover Telex and typos: `Buoi sang o Da nawngx` | `Buổi sáng ở Đà Nẵng` |

Splits are assigned by a hash of the sentence, so the same sentence can never appear in both train and test. A third test set, **hard**, keeps clean test sentences dense in ambiguous syllables (see [docs/experiment_plan.md](docs/experiment_plan.md)).

## Metrics

- **CER / WER** — edit distance over characters / words (robust to the length changes noisy input causes)
- **Word accuracy** and **sentence exact match** (of the default, greedy output)
- **EM@k** (k = 1, 3, 5, `eval --nbest 5`) — the reference is among the top-k candidates, as in an IME's candidate bar. LLMs use beam search (`num_beams = k`), the lexicon a beam over its bigram scores; EM@1 is the beam's top candidate, so it can differ from sentence exact match. GGUF models have no n-best yet (llama-cpp-python has no beam search)
- **Latency** — single-sentence p50 / p95, measured the way an IME calls the model (batch size 1, after warm-up)
- **Memory** — resident set size of the process

## Layout

```
src/virestore/
  text.py         normalisation, diacritic stripping, sentence splitting
  noise.py        Telex / typo noise generator
  data.py         corpus → train/dev/test JSONL
  hardset.py      test subset of sentences dense in ambiguous syllables
  metrics.py      CER, WER, word and sentence accuracy, EM@k
  prompts.py      one prompt format for training and inference
  tagger.py       character-level diacritic tagger (one encoder pass per sentence) + training
  train_tcn.py    retrain the acausal TCN baseline on this project's data
  models/         identity, lexicon, TCN, tagger, Hugging Face LLM, GGUF (llama.cpp, optional lexicon draft)
  eval.py         quality + latency + memory harness → results/
  analysis.py     error analysis of saved predictions (fidelity, confusions, CER by length)
  latency_profile.py  where a GGUF restoration spends its time (prompt, per-token, n-token forward)
  train_lora.py   LoRA / QLoRA SFT (loss on the answer only)
  merge.py        merge adapter → full model for conversion
  serve.py        FastAPI server for VIMEK
configs/          experiment configs
notebooks/        marimo notebook that runs the experiments on molab
docs/             experiment plan, quantisation, latency, error analysis, molab guide
data/sample/      tiny hand-written corpus for tests
tests/            unit tests (pytest)
```

## License

Code: [MIT](LICENSE). Training data: WanJuan-Vietnamese, CC BY 4.0 — see [data/README.md](data/README.md).
