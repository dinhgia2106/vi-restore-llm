# Running the experiments on molab

The GPU experiments run in a [marimo](https://marimo.io) notebook on a molab sandbox. The
notebook source is [`notebooks/experiments.py`](../notebooks/experiments.py); while a session is
live, the copy in the sandbox (`/marimo/notebook.py`) is the source of truth and the repo copy is
synced from it after each step.

## What the notebook does

1. Clones this repo into `/marimo/vi-restore-llm` (or `git pull`s it), so every run uses committed code.
   The commit hash is shown in the `repo` cell.
2. Downloads the WanJuan-Vietnamese shards listed in [data/README.md](../data/README.md) from the
   Hugging Face Hub with `snapshot_download`.
3. Runs the `virestore` CLIs (`data`, `hardset`, `eval`, later `train_lora`, `merge`) as
   **background jobs** and shows their status.
4. Shows `results/summary.md`.

## Background jobs

Long commands cannot run inside a cell: the connection to the sandbox drops after a while and
interrupts the cell (a first data build was killed this way). The `repo` cell defines:

- `launch(name, *steps)` — runs the steps one after another in a detached process; a list step is
  `python -m <step> ...`, a string step is a raw shell command (used for building llama.cpp and
  GGUF conversion). Output goes to `logs/<name>.log`, the exit code to `logs/<name>.exit`.
- `job_state(name)` — `not started`, `running`, `done` or `failed` (a finished child that was not
  reaped shows up as a zombie; this is reported as `failed`, not `running`).
- `reset_job(name)` — deletes the job's files so `launch` runs it again.

`launch` only starts a job that was never started, so re-running cells is safe and a failed job is
never retried silently. Cells that depend on a job call `mo.stop(...)` until it is `done`; cells
that reference `refresh` (a `mo.ui.refresh`, 30 s) re-check automatically while the notebook is
open in a browser.

Jobs: `data_clean`, `data_noisy`, `hard`, `e0`, `e2`, `e3_train`, `e3_eval`, `analysis`,
`llamacpp_build`, `e4_convert`, `e4_eval`, `gguf_upload`, ...

## Results flow

Runs write `results/<tag>.json` and regenerate `results/summary.md` in the sandbox checkout. Copy
them into the local repo and commit there; never edit `results/` by hand. Because the files are
tracked, a later `git pull` in the sandbox refuses to overwrite the sandbox's own untracked copies
of the same files, and a regenerated `summary.md` counts as a local modification. Before pulling
in the sandbox: for each incoming `results/` file, compare the sandbox copy with
`git show origin/main:<file>` (`cmp`) and remove it if identical; then
`git checkout -- results/summary.md` (it is regenerated from the JSON files on the next eval) and
`git pull --ff-only`. The `repo` cell pulls with `check=True`, so a blocked pull stops the notebook.

## Model files

Checkpoints and GGUF files are not committed. Adapters are pushed to **private** Hugging Face model
repos under the user's account (only with the user's approval), so they survive a sandbox reset:

- E3 adapter: `dinhgia2106/vi-restore-llm-qwen05-lora` (adapter, tokenizer, `train_config.json`;
  intermediate checkpoints left out).

## Secrets

Set `HF_TOKEN` in the kernel's environment (`os.environ["HF_TOKEN"] = ...` from a scratch cell or
the sandbox settings). Never write it into a cell: the notebook file is committed.

## Environment pitfalls

- The sandbox ships `torch 2.11.0+cu130` and a matching `torchvision`. Adding packages through
  marimo installs into an overlay venv (`/tmp/uv-venv`); adding `peft` / `accelerate` there pulled
  in `torch 2.15` from PyPI, which shadowed the system one and broke `torchvision`
  (`operator torchvision::nms does not exist`, surfacing as
  `Could not import module 'Qwen2ForCausalLM'`). Fix: remove `torch` from the overlay so the system
  build is used, then check `torch.__version__` and `torch.cuda.is_available()`.
- transformers 5 removed `torch_dtype=`; the code uses `dtype=` (transformers ≥ 4.56).
- `/proc/cpuinfo` is virtualised (gVisor), so results record `cpu: unknown` there.

## Latency

Quality numbers from the sandbox are valid. **Latency numbers from the sandbox are not the
50 ms CPU budget**: the GPU runs use CUDA, and the sandbox CPU is a virtualised server CPU. Every
result records `machine.device` and `machine.cpu_count`; the CPU latency of the final models is
measured on the target laptop.
