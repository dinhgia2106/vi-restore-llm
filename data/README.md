# Data

## Source corpus: WanJuan-Vietnamese (WanJuan3.0 / 万卷·丝路)

- **What:** the Vietnamese subset of WanJuan3.0, a cleaned web-text corpus released by OpenDataLab (Shanghai AI Laboratory) — about 81 M documents, 280 GB, 74 B tokens, JSON Lines.
- **Where:** https://opendatalab.com/OpenDataLab/WanJuan-Vietnamese (an OpenDataLab login is required to download) · project page https://github.com/opendatalab/WanJuan3.0
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
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed --limit 2000000
python -m virestore.data --corpus data/raw/wanjuan-vietnamese --out data/processed_noisy --noisy --limit 2000000
```

Start with a few shards and `--limit`; the full subset is far more than the experiments need.
