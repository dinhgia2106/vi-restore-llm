# Compression: fitting a larger tagger into the latency budget

The large tagger is the most accurate model in this project, but as float it needs too many CPU
threads for an input method. This study compares two ways of making it cheaper: **knowledge
distillation** into smaller architectures, and **int8 dynamic quantisation** of the ONNX graph.

All models are trained on the document split. *clean* has 17 841 test sentences, *external*
(held-out shards) 14 000.

## Models

| model | `d_model` × layers | heads | feed-forward | parameters | config |
|---|---|---|---|---|---|
| large tagger (teacher) | 512 × 12 | 8 | 2048 | 38.1M | `configs/tagger_large_ds.yaml` |
| tagger | 384 × 8 | 6 | 1536 | 14.4M | `configs/tagger_ds.yaml` |
| distilled tagger | 384 × 8 | 6 | 1536 | 14.4M | `configs/tagger_kd_ds.yaml` |
| small tagger | 256 × 6 | 4 | 1024 | 4.88M | `configs/tagger_s_ds.yaml` |
| small distilled tagger | 256 × 6 | 4 | 1024 | 4.88M | `configs/tagger_s_kd_ds.yaml` |

Distillation loss: `(1 - a) * CE + a * T^2 * KL(teacher || student)` with `a = 0.5`, `T = 2`, on the
same labelled characters as the cross-entropy.

## Quality

Sentence accuracy equals EM@1 for the tagger. Files: `results/ds-tagger*-{clean,external}.json`.

| model | clean CER | clean sentence acc | external CER | external sentence acc |
|---|---|---|---|---|
| large tagger | 0.0071 | 0.6634 | 0.0065 | 0.6691 |
| **large tagger, int8** | **0.0071** | **0.6625** | **0.0065** | **0.6691** |
| distilled tagger | 0.0091 | 0.5900 | 0.0085 | 0.6019 |
| distilled tagger, int8 | 0.0091 | 0.5902 | 0.0085 | 0.6026 |
| tagger | 0.0093 | 0.5811 | 0.0088 | 0.5903 |
| small distilled tagger | 0.0129 | 0.4733 | 0.0122 | 0.4861 |
| small distilled tagger, int8 | 0.0129 | 0.4726 | 0.0122 | 0.4859 |
| small tagger | 0.0130 | 0.4712 | 0.0124 | 0.4796 |

## Latency on the target laptop

Windows 11, Intel CPU with 32 logical processors, on AC power, Balanced plan, onnxruntime, batch size
1, first 200 clean sentences. Each configuration ran in five interleaved rounds with a PyTorch
tagger control (`results/laptop-ac-int8-b/`). The control's median p95 was 18.75 ms (range
16.50–21.15).

| model | threads | median p50 ms | median p95 ms | p95 range ms |
|---|---|---|---|---|
| large tagger, int8 | 1 | 28.43 | 48.23 | 46.90–49.70 |
| large tagger, int8 | 2 | 18.59 | 29.88 | 29.04–30.78 |
| **large tagger, int8** | **4** | **13.99** | **22.48** | **21.77–23.53** |
| large tagger, int8 | 8 | 12.72 | 19.33 | 19.03–19.99 |
| distilled tagger, int8 | 1 | 11.78 | 20.79 | 20.42–20.88 |
| distilled tagger, int8 | 2 | 8.17 | 13.63 | 13.30–13.97 |
| distilled tagger, int8 | 4 | 6.21 | 9.62 | 9.29–10.20 |
| distilled tagger, int8 | 8 | 5.95 | 9.09 | 8.73–11.98 |

An earlier three-round scan ran while the machine was busier: its control was slower (median p95
24.76 ms) and the large int8 model at 4 threads reached 36.99 ms (`results/laptop-ac-int8/`). It is
kept in the repository. **Latency on this laptop moves with background load.**

Float against int8 at 4 threads, three interleaved rounds (`results/laptop-ac-compress/`, control
median p95 18.22 ms):

| model | ONNX size | median p95 ms | int8 size | int8 median p95 ms |
|---|---|---|---|---|
| large tagger | 146.4 MB | 70.74 | 38.3 MB | 22.83 |
| distilled tagger | 55.7 MB | 29.66 | 15.1 MB | 10.00 |
| small distilled tagger | 19.1 MB | 11.10 | 5.6 MB | 4.70 |

## Findings

1. **Int8 quantisation is nearly free here.** Sentence accuracy moves by at most 0.09 points on these
   test sets, while the ONNX graph becomes 2.4–3.1× faster (p95, 4 threads) and 3.4–3.8× smaller.
2. **The large tagger fits the budget once quantised.** In the five-round scan its int8 graph stays
   under 50 ms at every thread count tried, including one thread (48.23 ms, with little margin), and
   at 4 threads the median p95 is 22.48 ms. Under the heavier background load of the earlier scan, one
   thread reached 103.89 ms and two threads 58.26 ms, so **use at least 4 threads**.
3. **Distillation helps a little.** At 14.4M parameters it adds 0.89 points of sentence accuracy on
   clean and 1.16 on external. At 4.88M it adds 0.21 and 0.65.
4. **Shrinking the architecture costs more than quantising it.** The distilled 14.4M model is 7.3
   points below the large tagger on clean, and the 4.88M model 19 points, while int8 costs 0.09.

## Recommendation

- **Default:** large tagger, int8 (`tagger.int8.onnx`), 4 threads.
- **Weak machines:** distilled tagger, int8, which needs less than half the time at every thread count.

## Caveats

- The exported graphs take **one sentence per call** (input shape `[1, chars]`). This is what an IME
  needs. Batched scoring of an ONNX model has to call it one sentence at a time.
- Teacher training time is not included in the cost of a distilled model.
- Training times are not reported as a comparison because the jobs shared one GPU at varying load.
- Latency was measured on one laptop only.
