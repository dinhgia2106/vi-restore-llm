# Data

## Source corpus: WanJuan-Vietnamese (WanJuan3.0 / 万卷·丝路)

- **What:** the Vietnamese subset of WanJuan3.0, a cleaned web-text corpus released by OpenDataLab (Shanghai AI Laboratory) — about 81 M documents, 280 GB, 74 B tokens, JSON Lines.
- **Where:** https://huggingface.co/datasets/opendatalab/WanJuan-Vietnamese (47 `jsonl.gz` shards, ~131 GB compressed; not gated) · also https://opendatalab.com/OpenDataLab/WanJuan-Vietnamese · project page https://github.com/opendatalab/WanJuan3.0
- **Schema:** one JSON object per line with `id`, `language`, `content` (the text), `labels`, `sub_path`, `title`. Some shards repeat the same record on consecutive lines; sentence-level de-duplication removes these.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The corpus is not redistributed here; only derived sentence pairs are produced locally.
- **Citation:**

```bibtex
@misc{yu2025wanjuansilu,
  title         = {WanJuanSiLu: A High-Quality Open-Source Webtext Dataset for Low-Resource Languages},
  author        = {Jia Yu and Fei Yuan and Rui Min and Jing Yu and Pei Chu and Jiayang Li and Wei Li and Ruijie Zhang and Zhenxiang Li and Zhifei Ren and Dong Zheng and Wenjian Zhang and Yan Teng and Lingyu Meng and ZhenJiang Jin and Jiantao Qiu and ShaSha Wang and Zhongying Tu and Dahua Lin and Yu Wang and Yu Qiao and Yanfeng Wang and Conghui He},
  year          = {2025},
  eprint        = {2501.14506},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2501.14506}
}
```

### Changes made to the original data

`python -m virestore.data` derives training pairs from the corpus. It:

1. splits documents into sentences and keeps those with 3–40 words that contain at least one diacritic;
2. drops sentences containing URLs, e-mail addresses or markup, or with < 70 % letters or > 10 % digits;
3. removes exact duplicates;
4. builds the model input by stripping diacritics (and, with `--noisy`, adding synthetic Telex/typo noise);
5. assigns train / dev / test (98 / 1 / 1) by a hash of the sentence.

### Shards used

One shard per top-level domain, taken from the `general` source of each domain. `culture/ebooks` is left out because it holds full translated books.

```
raw/culture/general/part-677f75d865d8-000359.jsonl.gz
raw/encyclopedia/general/part-677f75d865d8-000109.jsonl.gz
raw/general/general/part-677f75d865d8-000300.jsonl.gz
raw/history_policy/general/part-677f75d865d8-000270.jsonl.gz
raw/local_life/general/part-677f75d865d8-000393.jsonl.gz
raw/news/general/part-677f75d865d8-000285.jsonl.gz
raw/professional_field/general/part-677f75d865d8-000332.jsonl.gz
```

Documents are read round-robin across shards, so `--limit` samples every domain.

## Layout

```
data/
  raw/wanjuan-vietnamese/   downloaded *.jsonl(.gz) shards — not committed
  processed/                clean task pairs — not committed
  processed_noisy/          noisy task pairs — not committed
  sample/sample.txt         18 hand-written sentences used by the tests
```

## Building

```bash
hf download opendatalab/WanJuan-Vietnamese --repo-type dataset --local-dir data/raw/wanjuan-vietnamese --include "raw/news/general/part-677f75d865d8-000285.jsonl.gz"   # repeat per shard
```

```bash
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed --limit 2000000
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed_noisy --noisy --limit 2000000
```

Start with a few shards and `--limit`; the full subset is far more than the experiments need.
