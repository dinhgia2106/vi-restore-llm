# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.16.0",
#     "peft==0.22.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # vi-restore-llm: experiments

    Which restores Vietnamese diacritics best under a **50 ms p95 single-sentence CPU budget**:
    a frequency lexicon, an acausal TCN, or a LoRA-tuned small LLM, and what does int4 quantisation cost?

    This notebook drives the `virestore` CLIs from <https://github.com/dinhgia2106/vi-restore-llm>
    (plan: `docs/experiment_plan.md`, experiments E0-E6). Every number shown here comes from a
    `virestore.eval` run saved under `results/`.

    **Setup:** set `HF_TOKEN` in the environment before running; it is never written into this notebook.
    """)
    return


@app.cell(hide_code=True)
def job_refresh(mo):
    refresh = mo.ui.refresh(default_interval="30s", label="job status")
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def job_status(LOGS, job_state, mo, refresh):
    refresh
    _rows = []
    for _log in sorted(LOGS.glob("*.log")):
        _name = _log.stem
        _tail = _log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] or [""]
        _rows.append({"job": _name, "state": job_state(_name), "last log line": _tail[0][-160:]})
    mo.ui.table(_rows, selection=None) if _rows else mo.md("no jobs yet")
    return


@app.cell(hide_code=True)
def results_summary(REPO, mo, refresh):
    refresh
    _summary = REPO / "results/summary.md"
    mo.md(_summary.read_text(encoding="utf-8")) if _summary.exists() else mo.md("no results yet")
    return


@app.cell
def imports():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, os, subprocess, sys


@app.cell
def repo(Path, mo, os, subprocess, sys):
    REPO = Path("/marimo/vi-restore-llm")
    if not REPO.exists():
        subprocess.run(["git", "clone", "https://github.com/dinhgia2106/vi-restore-llm", str(REPO)], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "-q"], check=True)
    COMMIT = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    LOGS = REPO / "logs"
    LOGS.mkdir(exist_ok=True)
    _ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "utf-8"}


    def vr(*args: str) -> str:
        """Run `python -m virestore.<cli> ...` from the repo checkout and return its stdout (short jobs only)."""
        r = subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=_ENV, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        return r.stdout


    def job_state(name: str) -> str:
        exit_file = LOGS / f"{name}.exit"
        if exit_file.exists():
            return "done" if exit_file.read_text().strip() == "0" else "failed"
        pid = LOGS / f"{name}.pid"
        if not pid.exists():
            return "not started"
        try:  # a finished child that nobody reaped is a zombie ("Z"), not a running job
            return "failed" if Path(f"/proc/{int(pid.read_text())}/stat").read_text().split()[2] == "Z" else "running"
        except (OSError, ValueError, IndexError):
            return "failed"


    def launch(name: str, *steps: list) -> str:
        """Start `python -m <step>` commands one after another in a detached process.

        Long jobs outlive this cell; `logs/<name>.log` has the output and `logs/<name>.exit`
        the exit code. Only a job that was never started is launched, so re-running cells
        is safe; call `reset_job(name)` to run a failed job again.
        """
        state = job_state(name)
        if state != "not started":
            return state
        chain = " && ".join(" ".join([sys.executable, "-m", *map(str, s)]) for s in steps)
        cmd = f"({chain}) > {LOGS / name}.log 2>&1; echo $? > {LOGS / name}.exit"
        p = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=_ENV, start_new_session=True)
        (LOGS / f"{name}.pid").write_text(str(p.pid))
        return "started"


    def reset_job(name: str) -> None:
        for ext in ("pid", "log", "exit"):
            (LOGS / f"{name}.{ext}").unlink(missing_ok=True)


    mo.md(f"repo at `{REPO}` | commit `{COMMIT}`")
    return LOGS, REPO, job_state, launch


@app.cell
def shards(REPO, mo):
    from huggingface_hub import snapshot_download

    RAW = REPO / "data/raw/wanjuan-vietnamese"
    # one shard per top-level domain, `general` source; culture/ebooks left out (full translated books)
    SHARDS = [
        "raw/culture/general/part-677f75d865d8-000359.jsonl.gz",
        "raw/encyclopedia/general/part-677f75d865d8-000109.jsonl.gz",
        "raw/general/general/part-677f75d865d8-000300.jsonl.gz",
        "raw/history_policy/general/part-677f75d865d8-000270.jsonl.gz",
        "raw/local_life/general/part-677f75d865d8-000393.jsonl.gz",
        "raw/news/general/part-677f75d865d8-000285.jsonl.gz",
        "raw/professional_field/general/part-677f75d865d8-000332.jsonl.gz",
    ]
    snapshot_download("opendatalab/WanJuan-Vietnamese", repo_type="dataset", local_dir=RAW, allow_patterns=SHARDS, max_workers=8)
    mo.md("\n".join(f"- `{p}`: {(RAW / p).stat().st_size / 1e9:.2f} GB" for p in SHARDS))
    return (RAW,)


@app.cell(hide_code=True)
def data_md(mo):
    mo.md(r"""
    ## 1. Task data

    Sentence pairs (`input` = stripped / noised, `target` = original) from the shards above,
    `--limit 2000000` unique sentences, hash-based 98/1/1 split. The hard set keeps test sentences
    with at least two highly ambiguous syllables (see `virestore.hardset`).
    """)
    return


@app.cell
def build_data(RAW, REPO, job_state, launch, refresh):
    refresh
    LIMIT = 2_000_000
    CLEAN, NOISY = REPO / "data/processed", REPO / "data/processed_noisy"
    HARD = CLEAN / "test_hard.jsonl"
    DATA_JOBS = {
        "data_clean": launch("data_clean", ["virestore.data", "--corpus", RAW, "--out", CLEAN, "--limit", LIMIT]),
        "data_noisy": launch("data_noisy", ["virestore.data", "--corpus", RAW, "--out", NOISY, "--limit", LIMIT, "--noisy"]),
    }
    if job_state("data_clean") == "done":
        DATA_JOBS["hard"] = launch(
            "hard", ["virestore.hardset", "--train", CLEAN / "train.jsonl", "--test", CLEAN / "test.jsonl", "--out", HARD]
        )
    DATA_JOBS
    return CLEAN, HARD, NOISY


@app.cell
def data_sample(CLEAN, HARD, NOISY, job_state, json, mo, refresh):
    refresh
    mo.stop(
        not all(job_state(j) == "done" for j in ("data_clean", "data_noisy", "hard")),
        mo.md("waiting for the data jobs; re-run this cell when they are done"),
    )
    DATA_STATS = {
        "clean": json.loads((CLEAN / "stats.json").read_text())["counts"],
        "noisy": json.loads((NOISY / "stats.json").read_text())["counts"],
        "hard": sum(1 for _ in (HARD).open(encoding="utf-8")),
    }


    def _head(path, n=5):
        with path.open(encoding="utf-8") as f:
            return [json.loads(next(f)) for _ in range(n)]


    mo.vstack([
        DATA_STATS,
        mo.ui.tabs({
            "clean": mo.ui.table(_head(CLEAN / "test.jsonl"), selection=None),
            "noisy": mo.ui.table(_head(NOISY / "test.jsonl"), selection=None),
            "hard": mo.ui.table(_head(HARD), selection=None),
        }),
    ])
    return


@app.cell(hide_code=True)
def e0_md(mo):
    mo.md(r"""
    ## 2. E0: floors (identity, lexicon)

    `identity` returns the input unchanged; `lexicon` picks the most frequent accented form of each
    syllable seen in training (trained on the train split matching the test set). The hard set is
    drawn from the clean test split.
    """)
    return


@app.cell
def e0(CLEAN, HARD, NOISY, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(
        not all(job_state(j) == "done" for j in ("data_clean", "data_noisy", "hard")),
        mo.md("E0 waits for the data jobs"),
    )
    RESULTS = REPO / "results"
    E0_SETS = {
        "clean": (CLEAN / "train.jsonl", CLEAN / "test.jsonl"),
        "noisy": (NOISY / "train.jsonl", NOISY / "test.jsonl"),
        "hard": (CLEAN / "train.jsonl", HARD),
    }
    E0_JOB = launch(
        "e0",
        *[
            ["virestore.eval", "--model", m, "--train", tr, "--test", te, "--tag", f"e0-{m}-{s}", "--results", RESULTS, "--save-predictions"]
            for s, (tr, te) in E0_SETS.items()
            for m in ("identity", "lexicon")
        ],
    )
    E0_JOB
    return


@app.cell(hide_code=True)
def e2_md(mo):
    mo.md(r"""
    ## 3. E2: Qwen2.5-0.5B-Instruct zero-shot

    First 1k sentences of the clean test split, greedy decoding, prompt from `virestore.prompts`.
    Run on the sandbox GPU, so its latency is **not** the CPU budget number; CPU latency is measured
    later on the target laptop.
    """)
    return


@app.cell
def e2(CLEAN, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("data_clean") != "done", mo.md("E2 waits for the clean data job"))
    E2_JOB = launch(
        "e2",
        ["virestore.eval", "--model", "hf", "--hf-model", "Qwen/Qwen2.5-0.5B-Instruct", "--test", CLEAN / "test.jsonl",
         "--limit", 1000, "--batch-size", 32, "--latency-n", 100, "--tag", "e2-qwen05-zeroshot-clean",
         "--results", REPO / "results", "--save-predictions"],
    )
    E2_JOB
    return


if __name__ == "__main__":
    app.run()
