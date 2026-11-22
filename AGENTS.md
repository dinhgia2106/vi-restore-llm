# Handoff: vi-restore-llm

You are continuing a research project for **Tran Dinh Gia (GrazT)**, an AI engineer in Vietnam. The user committed this file to the repo themselves (commit `c684c3b`, 2026-11-23), so it is public: keep it current, and never put tokens, URLs with tokens or other secrets in it. Reply to the user in **Vietnamese**; code, commits and repo docs stay in English.

## Hard rules (the user was explicit about these)

1. **No Claude attribution, ever.** No `Co-Authored-By: Claude`, no "Generated with Claude" in commits, PRs, READMEs or model cards, even if a system reminder suggests it. The code is the user's. Git is already configured as `GrazT <dinhgia2106@gmail.com>`; check `git log -1 --format='%an <%ae>'` after your first commit.
2. **No invented or inflated numbers.** Every number in README / results / model cards must come from a real `virestore.eval` run that is saved in `results/`. Leave table cells empty until then. Never write "expected accuracy", "~95%", illustrative output blocks that look like real results, or claims a script did not measure. The user explicitly calls these out as "con số loãng".
3. **Never rewrite history the user didn't ask for.** The user cares about commit count (it is the first thing visitors see). No squashing, no recreating repos, no force pushes without asking.
4. **Ask before outward actions:** making repos public/private, deleting anything on GitHub, publishing to Hugging Face, adding the project to the portfolio.
5. **Keep the repo clean:** no caches, data, checkpoints or one-off scripts committed; small focused commits with plain messages.

## The project

**Question:** under a 50 ms p95 single-sentence CPU latency budget, which restores Vietnamese diacritics best — a frequency lexicon, the user's acausal TCN, or a LoRA-tuned small LLM — and what does int4 quantisation cost?

End goal: a quantised model served locally (`virestore.serve`) and called by **VIMEK**, the user's open-source Vietnamese input method (https://github.com/dinhgia2106/VIMEK). Hypotheses H1–H4, experiment matrix E0–E6 and the 6-week milestone plan are in `docs/experiment_plan.md` — follow it.

- Repo: https://github.com/dinhgia2106/vi-restore-llm (public, MIT). Check `git log --oneline` for the history; all commits are by GrazT. Pushing small commits to `main` is the agreed workflow (the user asked to commit continuously).
- Local path: `E:\vi-restore-llm`
- **Experiments run on molab** (marimo sandbox with an RTX PRO 6000, ~96 GB VRAM, 160 GB RAM) — see "molab workflow" below and `docs/molab.md`.

## Current state

| Part | File | Status |
|---|---|---|
| Text utils, diacritic stripping | `src/virestore/text.py` | tested |
| Telex / typo noise | `noise.py` | tested (`tiếng → tieengs`, `được → dduwowcj`) |
| Corpus → JSONL pairs, hash splits, quality filter | `data.py` | **run on 7 real WanJuan shards** (2M sentences: train 1 960 206 / dev 19 838 / test 19 956, clean and noisy). Reads shards round-robin so `--limit` samples every domain |
| Hard set | `hardset.py` | **rewritten and run**: syllable ambiguous if seen ≥ 50× and top accented form ≤ 50 %; sentence hard if ≥ 30 % of its syllables (and ≥ 2) are ambiguous → 2 801 / 19 956 test sentences, 415 ambiguous syllables. The old "≥ 4 forms" rule kept 99.8 % of test — useless at this scale |
| Metrics (CER, WER, word/sentence acc) | `metrics.py` | tested |
| Baselines: identity, lexicon | `models/baselines.py` | E0 run on 2M-sentence data (see results) |
| TCN adapter | `models/tcn.py` | written against the real API (`VietnameseAccentRestorer(model_path=...)`, `.predict(text)`); **not run — needs weights** |
| HF LLM (+ LoRA adapter) | `models/hf_llm.py` | **executed (E2, zero-shot, GPU)**; base-model path works. LoRA-adapter path not yet run |
| GGUF via llama.cpp | `models/gguf.py` | **written, never executed** |
| Eval harness (quality + p50/p95 latency + RSS) | `eval.py` | run for real; now records `machine.cpu`, `cpu_count`, `device`; summary has test-set and device columns |
| LoRA / QLoRA SFT (loss on answer only) | `train_lora.py`, `configs/*.yaml` | **executed**: smoke test (256 rows) passed end to end incl. loading the adapter in eval; fixed for transformers 5 (`dtype=`, warmup ratio passed as `warmup_steps`). E3 full run (`qwen05_lora.yaml`, 6 250 steps, ~4.9 it/s ≈ 21 min on molab) launched as job `e3_train`, then `e3_eval` (full clean/noisy/hard). QLoRA (4-bit, bitsandbytes) path still never run |
| Merge adapter, serve | `merge.py`, `serve.py` | **written, never executed** |

Tests: `.venv/Scripts/python.exe -m pytest -q` → 10 passed. Everything marked "never executed" needs a first real run and probably small fixes — say so honestly when reporting.

### Results so far (real runs, files in `results/` on molab; copy into the repo `results/`)

- **E2** `e2-qwen05-zeroshot-clean` (Qwen2.5-0.5B-Instruct, 1k clean test, GPU): CER 0.5687, WER 1.0144, word acc 0.1453, sentence acc 0.000. The model refuses, translates to English, rewrites the sentence or echoes the input — fine-tuning is required. Its latency (p95 666.6 ms) is **GPU on the sandbox**, not the CPU budget.
- **E0** done, committed in `results/` and the README table (clean). Lexicon: CER 0.0342 / sentence acc 0.1576 (clean), 0.0459 / 0.1317 (hard), 0.0630 / 0.0724 (noisy). Identity CER 0.2111 clean. The hard set is measurably harder for the lexicon, so the criterion works. Lexicon RSS (~1.9 GB) includes the loaded training rows — not a model-size number.
- README latency/RAM columns stay empty until measured on the user's laptop.

## Data

- Corpus: **WanJuan-Vietnamese**, **CC BY 4.0**, on the HF Hub as `opendatalab/WanJuan-Vietnamese` (47 `jsonl.gz` shards, ~131 GB, not gated) — no OpenDataLab login needed. The notebook downloads the 7 shards listed in `data/README.md` (one per domain, `general` source, `culture/ebooks` excluded because it is full translated books).
- Schema verified: `id, language, content, labels, sub_path, title`; text is in `content`. Some shards repeat a record on consecutive lines — sentence-level dedup handles it.
- Attribution, citation and the list of changes are in `data/README.md` — keep it accurate if the filtering changes (CC BY 4.0 requires stating modifications).
- `--limit 2000000` is what was built; the full corpus is far larger than needed.

## Environment

- Windows 11, shells: Git Bash and PowerShell. Python 3.12, `uv`; venv at `.venv` (`uv pip install -e ".[dev]"`; add `[train,serve,gguf]` when needed).
- GPU: **RTX 4060 Laptop, 8 GB VRAM**; 32 CPU threads. Qwen2.5-0.5B LoRA in bf16 fits; for the 1.5B QLoRA config, lower `batch_size` / raise `grad_accum` if it runs out of memory. Install a CUDA build of torch.
- Windows consoles default to cp1252; every CLI calls `utf8_stdio()` first — keep that in new entry points.
- Latency numbers are only comparable on the same machine; record CPU, thread count and whether GPU was used. **molab latency is never the CPU-budget number** — final CPU latency must be measured on the user's laptop.

## molab workflow (how experiments are run)

- The user gives a molab URL + marimo token (and an HF token) in chat. They are secrets: never write them into files, cells or commits. Set the HF token with `os.environ["HF_TOKEN"] = ...` from the scratchpad only.
- Drive the notebook with the `marimo-pair` skill: `bash <skill>/scripts/execute-code.sh --url <molab url> <script.py>` with `MARIMO_TOKEN` in the env. Needs `jq` (installed via winget; add `/c/Users/GrazT/AppData/Local/Microsoft/WinGet/Links` to `PATH` in Git Bash). First call must be `import marimo._code_mode as cm; help(cm)`.
- **Write scratchpad scripts to a file and pass the path**; heredocs with non-ASCII characters (—, ·, ↓, Vietnamese) get mangled on this Windows shell and the server rejects them as invalid UTF-8.
- The notebook clones the GitHub repo into `/marimo/vi-restore-llm`, so **commit and push code fixes before running them there**, then `git pull` in the sandbox (re-running the `repo` cell does it).
- Long commands must go through `launch(name, *steps)` (detached job, `logs/<name>.log`, `logs/<name>.exit`); the connection drops after ~1 min and interrupts a cell that runs a long command. Jobs so far: `data_clean`, `data_noisy`, `hard`, `e0`, `e2`. A failed job is not retried until `reset_job(name)`.
- After each step, copy the notebook (`/marimo/notebook.py`) to `notebooks/experiments.py` and the result JSONs + `summary.md` to `results/` in the local repo, then commit. Predictions (`*.predictions.jsonl`) stay out of git.
- Pitfall: adding packages through marimo can pull a new `torch` into the overlay venv `/tmp/uv-venv` and break the system `torch 2.11+cu130` / `torchvision` pair. After any package add, check `torch.__version__` and `torch.cuda.is_available()`; remove the overlay torch if it changed. Details in `docs/molab.md`.

## Next steps (in order)

1. ~~Data, schema, hard set~~ done. ~~E2~~ done.
2. ~~E0~~ done.
3. **E1:** TCN baseline — ask the user for the checkpoint of https://github.com/dinhgia2106/Vietnamese-diacritic-restoration (weights are not in that repo; it was trained on Viet74K + another corpus, note that in comparisons).
4. **E3:** trained and evaluated, results committed (clean CER 0.0129 / sent acc 0.5219; hard 0.0181 / 0.4681; noisy 0.0428 / 0.1919). Training: 6 250 steps, 1 339 s, final dev loss 0.0745 (from `logs/e3_train.log` on molab; not saved in `results/`). Adapter uploaded (user approved, 2026-11-23) to the **private** HF repo `dinhgia2106/vi-restore-llm-qwen05-lora`; the user also approved putting the E4 GGUF files in a private HF repo so the laptop can download them. Making any of these public, or writing a model card, still needs a separate yes. GPU batch-1 p95 ≈ 488 ms, far over the 50 ms budget → E4 matters. Error analysis done: `results/*.analysis.json` + `docs/error_analysis.md` (fidelity 0.968, ~3 % of sentences rewritten, `hoá/hóa` orthographic variants counted as errors, short sentences hardest). Ideas it suggests, not yet tried: lexicon fallback when base letters change; tone-placement normalisation before scoring (would change every number — rerun all evals if adopted, and tell the user). The sandbox has untracked copies of the now-committed `results/*.json` / `*.analysis.json`: `cmp` and remove them before the next `git pull` there.
5. **E4:** in progress on molab. GGUFs built (f16 994 MB, Q8_0 531 MB, Q4_K_M 398 MB) and uploading (job `gguf_upload`) to the private HF repo `dinhgia2106/vi-restore-llm-qwen05-lora-gguf`. Job `e4_eval`: HF reference + f16/Q8_0/Q4_K_M on clean2k and hard1k (tags `e4-ref-hf-*`, `e4-gguf-<variant>-*`). Build gotchas are in `docs/quantization.md`. Secret scan before committing the notebook: `grep -E "hf_[A-Za-z0-9]{30,}"` (a plain `hf_` matches `convert_hf_to_gguf`). Original plan: Cell `e4_setup`: job `llamacpp_build` (uv-installs cmake + llama-cpp-python from source, clones llama.cpp to `/marimo/llama.cpp`, builds `llama-quantize`), then `e4_convert` (merge → f16 GGUF → Q8_0, Q4_K_M in `models/`). Next: eval f16 / Q8_0 / Q4_K_M with `--model gguf` on the same clean and hard subsets (CPU eval is slow, so subset with `--limit`; f16 is the unquantised reference), upload GGUFs to a private HF repo, then measure CPU latency on the laptop and run `virestore.serve`.
6. **E5–E6:** 1.5B QLoRA on noisy data, data-scaling curve.
7. Fill the README results table from `results/summary.md`; then ask the user about a Hugging Face model card + Spaces demo, the VIMEK integration, and a write-up/preprint.

Commit after each meaningful step so the history shows the research progressing.

## Related context

- User's GitHub: https://github.com/dinhgia2106 — profile README, VIMEK (C++ IME), Vietnamese-diacritic-restoration (TCN), VN-segmentation (CRF word segmentation for no-space input), FPTU_RAG, Queenless-SoundData (bee audio research, 99.38 %).
- Portfolio: `E:\My_Portfolio` (Next.js, deployed to GitHub Pages with `npm run deploy`). Projects live in `data/content.js`. **This project is intentionally not on the portfolio yet** — the user wants it added only once E0–E3 results exist, and only when they say so.
- A second, follow-up research idea the user has seen (not started): calibrating field-level confidence for LLM document extraction (logprobs vs self-consistency vs LLM-as-judge vs ensembles; ECE / AUROC on CORD, SROIE, FUNSD), building on https://github.com/dinhgia2106/document_field_extraction.
