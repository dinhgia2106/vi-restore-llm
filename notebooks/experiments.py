# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.16.0",
#     "huggingface-hub==2.4.0",
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
        """Start `python -m <step>` commands (or raw shell strings) one after another in a detached process.

        Long jobs outlive this cell; `logs/<name>.log` has the output and `logs/<name>.exit`
        the exit code. Only a job that was never started is launched, so re-running cells
        is safe; call `reset_job(name)` to run a failed job again.
        """
        state = job_state(name)
        if state != "not started":
            return state
        chain = " && ".join(s if isinstance(s, str) else " ".join([sys.executable, "-m", *map(str, s)]) for s in steps)
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


@app.cell(hide_code=True)
def e3_md(mo):
    mo.md(r"""
    ## 4. E3: Qwen2.5-0.5B-Instruct + LoRA

    `configs/qwen05_lora.yaml`: LoRA r=16 on all attention and MLP projections, 200k clean training
    pairs, 1 epoch, loss on the answer only. Evaluated on the full clean, noisy and hard test sets
    (GPU, batch 64), so the numbers line up with E0.
    """)
    return


@app.cell
def e3(CLEAN, HARD, NOISY, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("hard") != "done", mo.md("E3 waits for the data jobs"))
    E3_ADAPTER = REPO / "runs/qwen05-lora"
    E3_JOBS = {"e3_train": launch("e3_train", ["virestore.train_lora", "--config", REPO / "configs/qwen05_lora.yaml"])}
    if job_state("e3_train") == "done":
        E3_JOBS["e3_eval"] = launch(
            "e3_eval",
            *[
                ["virestore.eval", "--model", "hf", "--hf-model", "Qwen/Qwen2.5-0.5B-Instruct", "--adapter", E3_ADAPTER,
                 "--test", te, "--batch-size", 64, "--latency-n", 100, "--tag", f"e3-qwen05-lora-{s}",
                 "--results", REPO / "results", "--save-predictions"]
                for s, te in {"clean": CLEAN / "test.jsonl", "noisy": NOISY / "test.jsonl", "hard": HARD}.items()
            ],
        )
    E3_JOBS
    return (E3_ADAPTER,)


@app.cell(hide_code=True)
def e4_md(mo):
    mo.md(r"""
    ## 5. E4: merge, GGUF, quantisation

    The E3 adapter is merged into the base model, converted to GGUF with llama.cpp and quantised to
    Q8_0 and Q4_K_M; every variant is then evaluated through `llama-cpp-python` on the CPU. Quality
    numbers are valid here; **latency must be re-measured on the target laptop**. Files go to
    `models/` (not committed). Adapter: private HF repo `dinhgia2106/vi-restore-llm-qwen05-lora`.
    """)
    return


@app.cell
def e4_setup(E3_ADAPTER, Path, REPO, job_state, launch, os, refresh, sys):
    refresh
    LLAMA_CPP = Path("/marimo/llama.cpp")
    MODELS = REPO / "models"
    MERGED = REPO / "runs/qwen05-lora-merged"
    E4_JOBS = {
        "llamacpp_build": launch(
            "llamacpp_build",
            f"uv pip install --python {sys.executable} cmake ninja llama-cpp-python sentencepiece",  # the sandbox has gcc but no make
            f"(test -d {LLAMA_CPP} || git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP})",
            f"cmake -S {LLAMA_CPP} -B {LLAMA_CPP}/build -G Ninja -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
            f"cmake --build {LLAMA_CPP}/build --target llama-quantize -j {os.cpu_count()}",
        )
    }
    if job_state("e3_train") == "done" and job_state("llamacpp_build") == "done":
        E4_JOBS["e4_convert"] = launch(
            "e4_convert",
            ["virestore.merge", "--base", "Qwen/Qwen2.5-0.5B-Instruct", "--adapter", E3_ADAPTER, "--out", MERGED],
            f"mkdir -p {MODELS}",
            # PYTHONSAFEPATH=1 in the sandbox, so the converter's own directory must be on the path
            f"PYTHONPATH={LLAMA_CPP}:{LLAMA_CPP}/gguf-py {sys.executable} {LLAMA_CPP}/convert_hf_to_gguf.py {MERGED} --outfile {MODELS}/qwen05-lora-f16.gguf --outtype f16",
            f"{LLAMA_CPP}/build/bin/llama-quantize {MODELS}/qwen05-lora-f16.gguf {MODELS}/qwen05-lora-q8_0.gguf Q8_0",
            f"{LLAMA_CPP}/build/bin/llama-quantize {MODELS}/qwen05-lora-f16.gguf {MODELS}/qwen05-lora-q4_k_m.gguf Q4_K_M",
        )
    E4_JOBS
    return (MODELS,)


@app.cell
def e4_eval(
    CLEAN,
    E3_ADAPTER,
    HARD,
    MODELS,
    REPO,
    job_state,
    launch,
    mo,
    refresh,
):
    refresh
    mo.stop(job_state("e4_convert") != "done", mo.md("E4 eval waits for the GGUF conversion"))
    # CPU decoding through llama.cpp is ~0.3 s per sentence here, so E4 uses fixed subsets; the HF model
    # (E3, GPU) is re-run on the same subsets as the unquantised reference.
    E4_SUBSETS = {"clean2k": (CLEAN / "test.jsonl", 2000), "hard1k": (HARD, 1000)}
    E4_VARIANTS = ["f16", "q8_0", "q4_k_m"]
    _steps = [
        ["virestore.eval", "--model", "hf", "--hf-model", "Qwen/Qwen2.5-0.5B-Instruct", "--adapter", E3_ADAPTER,
         "--test", te, "--limit", n, "--batch-size", 64, "--latency-n", 0, "--tag", f"e4-ref-hf-{s}",
         "--results", REPO / "results", "--save-predictions"]
        for s, (te, n) in E4_SUBSETS.items()
    ] + [
        ["virestore.eval", "--model", "gguf", "--gguf", MODELS / f"qwen05-lora-{v}.gguf", "--test", te, "--limit", n,
         "--latency-n", 100, "--tag", f"e4-gguf-{v}-{s}", "--results", REPO / "results", "--save-predictions"]
        for v in E4_VARIANTS
        for s, (te, n) in E4_SUBSETS.items()
    ]
    E4_EVAL_JOB = launch("e4_eval", *_steps)
    E4_EVAL_JOB
    return


@app.cell(hide_code=True)
def emk_md(mo):
    mo.md(r"""
    ## 6. EM@k: top-k candidates

    EM@k = the reference is among the first k candidates (k = 1, 3, 5), as in the candidate bar of an
    input method. The LLM uses beam search (`num_beams = 5`), the lexicon a beam over its bigram
    scores. Runs are tagged `-nbest`; the earlier greedy runs are left untouched.
    """)
    return


@app.cell
def emk(CLEAN, E3_ADAPTER, HARD, NOISY, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("e3_train") != "done", mo.md("EM@k waits for the E3 adapter"))
    _sets = {"clean": (CLEAN / "train.jsonl", CLEAN / "test.jsonl"), "noisy": (NOISY / "train.jsonl", NOISY / "test.jsonl"), "hard": (CLEAN / "train.jsonl", HARD)}
    _hf = ["--model", "hf", "--hf-model", "Qwen/Qwen2.5-0.5B-Instruct", "--adapter", E3_ADAPTER, "--batch-size", 32]
    EMK_JOB = launch(
        "emk",
        ["virestore.eval", *_hf, "--test", CLEAN / "dev.jsonl", "--limit", 64, "--nbest", 5, "--latency-n", 0,
         "--tag", "smoke-nbest", "--results", "/marimo/smoke/results"],
        *[
            ["virestore.eval", "--model", "lexicon", "--train", tr, "--test", te, "--nbest", 5, "--latency-n", 0,
             "--tag", f"e0-lexicon-{s}-nbest", "--results", REPO / "results", "--save-predictions"]
            for s, (tr, te) in _sets.items()
        ],
        *[
            ["virestore.eval", *_hf, "--test", te, "--nbest", 5, "--latency-n", 0,
             "--tag", f"e3-qwen05-lora-{s}-nbest", "--results", REPO / "results", "--save-predictions"]
            for s, (_, te) in _sets.items()
        ],
    )
    EMK_JOB
    return


@app.cell(hide_code=True)
def tagger_md(mo):
    mo.md(r"""
    ## 7. Character-level tagger (non-autoregressive)

    The LLM cannot meet the 50 ms CPU budget: on the target laptop the prompt alone costs about
    80 ms and every generated token about 10 ms (`results/profile-laptop-q4_k_m-t8.profile.json`).
    `virestore.tagger` instead labels every vowel of the input in one encoder pass
    (`configs/tagger.yaml`). Trained on the clean training split; evaluated like every other model.
    """)
    return


@app.cell
def tagger(CLEAN, HARD, NOISY, Path, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("hard") != "done", mo.md("tagger waits for the data jobs"))
    import yaml as _yaml

    TAGGER_DIR = REPO / "runs/tagger"
    _cfg = _yaml.safe_load((REPO / "configs/tagger.yaml").read_text())
    _cfg["output_dir"] = "/marimo/smoke/tagger"
    _cfg["data"].update(train_limit=20000, dev_limit=500)
    _cfg["train"].update(epochs=1, eval_steps=40, log_steps=20)
    Path("/marimo/smoke").mkdir(exist_ok=True)
    Path("/marimo/smoke/tagger.yaml").write_text(_yaml.safe_dump(_cfg))
    TAGGER_JOBS = {
        "tagger_train": launch(
            "tagger_train",
            ["virestore.tagger", "--config", "/marimo/smoke/tagger.yaml"],
            ["virestore.eval", "--model", "tagger", "--tagger", "/marimo/smoke/tagger", "--test", CLEAN / "dev.jsonl",
             "--limit", 200, "--latency-n", 0, "--tag", "smoke-tagger", "--results", "/marimo/smoke/results"],
            ["virestore.tagger", "--config", REPO / "configs/tagger.yaml"],
        )
    }
    if job_state("tagger_train") == "done":
        TAGGER_JOBS["tagger_eval"] = launch(
            "tagger_eval",
            *[
                ["virestore.eval", "--model", "tagger", "--tagger", TAGGER_DIR, "--test", te, "--batch-size", 256,
                 "--latency-n", 0, "--nbest", 5, "--tag", f"t2-tagger-{s}", "--results", REPO / "results", "--save-predictions"]
                for s, te in {"clean": CLEAN / "test.jsonl", "noisy": NOISY / "test.jsonl", "hard": HARD}.items()
            ],
        )
    TAGGER_JOBS
    return


@app.cell
def tagger_large(CLEAN, HARD, NOISY, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("data_clean") != "done", mo.md("the large tagger waits for the data"))
    TAGGER_LARGE_DIR = REPO / "runs/tagger-large"
    TAGGER_LARGE_JOBS = {"tagger_large_train": launch("tagger_large_train", ["virestore.tagger", "--config", REPO / "configs/tagger_large.yaml"])}
    if job_state("tagger_large_train") == "done":
        TAGGER_LARGE_JOBS["tagger_large_eval"] = launch(
            "tagger_large_eval",
            *[
                ["virestore.eval", "--model", "tagger", "--tagger", TAGGER_LARGE_DIR, "--test", te, "--batch-size", 256,
                 "--latency-n", 0, "--nbest", 5, "--tag", f"t2-tagger-large-{s}", "--results", REPO / "results", "--save-predictions"]
                for s, te in {"clean": CLEAN / "test.jsonl", "noisy": NOISY / "test.jsonl", "hard": HARD}.items()
            ],
        )
    TAGGER_LARGE_JOBS
    return


@app.cell(hide_code=True)
def scaling_md(mo):
    mo.md(r"""
    ## 7b. Tagger data scaling (E6 for the tagger)

    Same model as `configs/tagger.yaml`, trained on the first 50k / 200k / 1M clean training pairs for
    3 epochs, plus 200k pairs for 1 epoch - the data budget of the LoRA run (E3) - so the tagger and
    the LLM can be compared at equal data. Evaluated on the full clean and hard test sets.
    """)
    return


@app.cell
def tagger_scaling(CLEAN, HARD, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("tagger_train") != "done", mo.md("scaling waits for the first tagger"))
    _runs = {"50k-e3": (50_000, 3), "200k-e1": (200_000, 1), "200k-e3": (200_000, 3), "1m-e3": (1_000_000, 3)}
    _steps = []
    for _name, (_n, _ep) in _runs.items():
        _dir = REPO / f"runs/tagger-{_name}"
        _steps.append(["virestore.tagger", "--config", REPO / "configs/tagger.yaml", "--train-limit", _n, "--epochs", _ep, "--output-dir", _dir])
        for _s, _te in {"clean": CLEAN / "test.jsonl", "hard": HARD}.items():
            _steps.append(["virestore.eval", "--model", "tagger", "--tagger", _dir, "--test", _te, "--batch-size", 256,
                           "--latency-n", 0, "--tag", f"t6-tagger-{_name}-{_s}", "--results", REPO / "results"])
    SCALING_JOB = launch("tagger_scaling", *_steps)
    SCALING_JOB
    return


@app.cell(hide_code=True)
def e1_md(mo):
    mo.md(r"""
    ## 8. E1: acausal TCN, retrained

    The original checkpoint is lost, so the TCN is retrained with its own repo's code
    (`virestore.train_tcn`: architecture, dataset and trainer from
    <https://github.com/dinhgia2106/Vietnamese-diacritic-restoration>) on the clean training split.
    """)
    return


@app.cell
def e1(CLEAN, HARD, NOISY, Path, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("hard") != "done", mo.md("E1 waits for the data jobs"))
    TCN_REPO = Path("/marimo/Vietnamese-diacritic-restoration")
    TCN_DIR = REPO / "runs/tcn"
    TCN_JOBS = {
        "e1_train": launch(
            "e1_train",
            f"(test -d {TCN_REPO} || git clone --depth 1 https://github.com/dinhgia2106/Vietnamese-diacritic-restoration {TCN_REPO})",
            ["virestore.train_tcn", "--tcn-repo", TCN_REPO, "--train", CLEAN / "train.jsonl", "--dev", CLEAN / "dev.jsonl",
             "--out", TCN_DIR, "--epochs", 3, "--batch-size", 128,
             # 300k sentences: a full epoch over 1.96M took ~8 h on the shared GPU; the LoRA runs use 200k
             "--limit", 300000],
        )
    }
    if job_state("e1_train") == "done":
        TCN_JOBS["e1_eval"] = launch(
            "e1_eval",
            *[
                ["virestore.eval", "--model", "tcn", "--tcn-repo", TCN_REPO, "--tcn-weights", TCN_DIR / "best_model.pth",
                 "--test", te, "--latency-n", 0, "--tag", f"e1-tcn-{s}", "--results", REPO / "results", "--save-predictions"]
                for s, te in {"clean": CLEAN / "test.jsonl", "noisy": NOISY / "test.jsonl", "hard": HARD}.items()
            ],
        )
    TCN_JOBS
    return


@app.cell(hide_code=True)
def e5_md(mo):
    mo.md(r"""
    ## 9. E5: Qwen2.5-1.5B-Instruct + LoRA on noisy pairs

    `configs/qwen15_lora_noisy.yaml` (bf16 base on the sandbox GPU; otherwise as `qwen15_qlora.yaml`).
    """)
    return


@app.cell
def e5(CLEAN, HARD, NOISY, REPO, job_state, launch, mo, refresh):
    refresh
    mo.stop(job_state("hard") != "done", mo.md("E5 waits for the data jobs"))
    E5_ADAPTER = REPO / "runs/qwen15-lora-noisy"
    E5_JOBS = {"e5_train": launch("e5_train", ["virestore.train_lora", "--config", REPO / "configs/qwen15_lora_noisy.yaml"])}
    if job_state("e5_train") == "done":
        E5_JOBS["e5_eval"] = launch(
            "e5_eval",
            *[
                ["virestore.eval", "--model", "hf", "--hf-model", "Qwen/Qwen2.5-1.5B-Instruct", "--adapter", E5_ADAPTER,
                 "--test", te, "--batch-size", 64, "--latency-n", 0, "--tag", f"e5-qwen15-lora-noisy-{s}",
                 "--results", REPO / "results", "--save-predictions"]
                for s, te in {"clean": CLEAN / "test.jsonl", "noisy": NOISY / "test.jsonl", "hard": HARD}.items()
            ],
        )
    E5_JOBS
    return


@app.cell
def ckpt_sync(LOGS, launch, refresh, sys):
    refresh
    # Upload training checkpoints to private HF repos every 20 min while they train, so a sandbox
    # shutdown cannot take them with it (it did once, 2026-11-23).
    CKPT_REPOS = {"runs/tcn": "dinhgia2106/vi-restore-llm-tcn", "runs/tagger-large": "dinhgia2106/vi-restore-llm-tagger-large"}
    _sync = "; ".join(
        f"from huggingface_hub import HfApi; a = HfApi(); a.create_repo('{r}', private=True, exist_ok=True); "
        f"import os; os.path.isdir('{d}') and a.upload_folder(repo_id='{r}', folder_path='{d}')"
        for d, r in CKPT_REPOS.items()
    )
    CKPT_SYNC = launch(
        "ckpt_sync",
        "while true; do "
        f"{sys.executable} -c \"{_sync}\" || true; "
        f"test -f {LOGS}/e1_train.exit && test -f {LOGS}/tagger_large_train.exit && break; sleep 1200; done",
    )
    CKPT_SYNC
    return


@app.cell
def docsplit(LOGS, RAW, REPO, job_state, launch, mo, refresh, sys):
    refresh
    # Document-level split: rebuild the data, then retrain and score every trained system on it.
    mo.stop(
        not all(job_state(j) == "done" for j in ("e1_eval", "tagger_large_eval")),
        mo.md("the document-split runs wait for the TCN and large tagger evals"),
    )
    DS, DSN = REPO / "data/doc_split", REPO / "data/doc_split_noisy"
    _ext = REPO / "data/raw/wanjuan-external"
    _shards = [
        "raw/culture/common_crawl/part-677f75d865d8-001249.jsonl.gz",
        "raw/encyclopedia/common_crawl/part-677f75d865d8-001285.jsonl.gz",
        "raw/general/common_crawl/part-677f75d865d8-000073.jsonl.gz",
        "raw/history_policy/general/part-677f75d865d8-000358.jsonl.gz",
        "raw/local_life/general/part-677f75d865d8-000731.jsonl.gz",
        "raw/news/general/part-677f75d865d8-000509.jsonl.gz",
        "raw/professional_field/general/part-677f75d865d8-000530.jsonl.gz",
    ]
    _dl = (
        "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/WanJuan-Vietnamese', "
        f"repo_type='dataset', local_dir='{_ext}', allow_patterns={_shards!r}, max_workers=8)"
    )
    DS_JOBS = {
        "ds_data": launch(
            "ds_data",
            f'{sys.executable} -c "{_dl}"',
            ["virestore.data", "--corpus", RAW, "--out", DS, "--limit", 2000000, "--split-by", "document"],
            ["virestore.data", "--corpus", RAW, "--out", DSN, "--limit", 2000000, "--split-by", "document", "--noisy"],
            ["virestore.hardset", "--train", DS / "train.jsonl", "--test", DS / "test.jsonl", "--out", DS / "test_hard.jsonl"],
            ["virestore.docsplit", "--corpus", _ext, "--train", DS / "train.jsonl", "--out", DS],
            ["virestore.docsplit", "--corpus", _ext, "--train", DSN / "train.jsonl", "--out", DSN, "--noisy"],
        )
    }
    if job_state("ds_data") == "done":
        _sets = {
            "clean": DS / "test.jsonl", "hard": DS / "test_hard.jsonl", "noisy": DSN / "test.jsonl",
            "external": DS / "external.jsonl", "external-noisy": DSN / "external.jsonl",
        }
        _domains = ("culture", "encyclopedia", "general", "history_policy", "local_life", "news", "professional_field")

        def _evals(name, model_args, extra=()):
            return [
                ["virestore.eval", *model_args, "--test", te, "--latency-n", 0, *extra, "--tag", f"ds-{name}-{s}",
                 "--results", REPO / "results", "--save-predictions"]
                for s, te in _sets.items()
            ]

        _lex = [
            ["virestore.eval", "--model", "lexicon", "--train", (DSN if "noisy" in s else DS) / "train.jsonl", "--test", te,
             "--latency-n", 0, "--nbest", 5, "--tag", f"ds-lexicon-{s}", "--results", REPO / "results", "--save-predictions"]
            for s, te in _sets.items()
        ]
        _tagger = ["--model", "tagger", "--tagger", REPO / "runs/ds-tagger", "--batch-size", 256]
        _tcn = ["--model", "tcn", "--tcn-repo", "/marimo/Vietnamese-diacritic-restoration", "--tcn-weights", REPO / "runs/ds-tcn/best_model.pth"]
        _q05 = ["--model", "hf", "--hf-model", "Qwen/Qwen2.5-0.5B-Instruct", "--adapter", REPO / "runs/ds-qwen05-lora", "--batch-size", 64]
        _q15 = ["--model", "hf", "--hf-model", "Qwen/Qwen2.5-1.5B-Instruct", "--adapter", REPO / "runs/ds-qwen15-lora-noisy", "--batch-size", 64]
        DS_JOBS["ds_lexicon"] = launch("ds_lexicon", *_lex)
        DS_JOBS["ds_gpu1"] = launch(
            "ds_gpu1",
            ["virestore.tagger", "--config", REPO / "configs/tagger_ds.yaml"],
            *_evals("tagger", _tagger, ("--nbest", 5)),
            *[
                ["virestore.eval", *_tagger, "--test", DS / f"external_{d}.jsonl", "--latency-n", 0, "--nbest", 5,
                 "--tag", f"ds-tagger-external-{d}", "--results", REPO / "results"]
                for d in _domains
            ],
            ["virestore.train_lora", "--config", REPO / "configs/qwen05_lora_ds.yaml"],
            *_evals("qwen05-lora", _q05),
        )
        DS_JOBS["ds_gpu2"] = launch(
            "ds_gpu2",
            ["virestore.train_tcn", "--tcn-repo", "/marimo/Vietnamese-diacritic-restoration", "--train", DS / "train.jsonl",
             "--dev", DS / "dev.jsonl", "--out", REPO / "runs/ds-tcn", "--epochs", 3, "--batch-size", 128, "--limit", 300000],
            *_evals("tcn", _tcn),
            ["virestore.train_lora", "--config", REPO / "configs/qwen15_lora_noisy_ds.yaml"],
            *_evals("qwen15-lora-noisy", _q15),
        )
        DS_JOBS["ds_gpu3"] = launch(
            "ds_gpu3",
            ["virestore.tagger", "--config", REPO / "configs/tagger_large_ds.yaml"],
            *_evals("tagger-large", ["--model", "tagger", "--tagger", REPO / "runs/ds-tagger-large", "--batch-size", 256], ("--nbest", 5)),
            *[
                ["virestore.eval", "--model", "tagger", "--tagger", REPO / "runs/ds-tagger-large", "--batch-size", 256,
                 "--test", DS / f"external_{d}.jsonl", "--latency-n", 0, "--nbest", 5,
                 "--tag", f"ds-tagger-large-external-{d}", "--results", REPO / "results"]
                for d in _domains
            ],
        )
        _up = "from huggingface_hub import HfApi; import os; a = HfApi(); a.create_repo('dinhgia2106/vi-restore-llm-docsplit', private=True, exist_ok=True); " + "; ".join(
            f"os.path.isdir('runs/{d}') and a.upload_folder(repo_id='dinhgia2106/vi-restore-llm-docsplit', folder_path='runs/{d}', "
            f"path_in_repo='{d}', ignore_patterns=['*/optimizer.pt', '*/scheduler.pt', '*/rng_state*'])"
            for d in ("ds-tagger", "ds-tagger-large", "ds-tcn", "ds-qwen05-lora", "ds-qwen15-lora-noisy")
        )
        # the exit check comes before the upload, so the last pass always sees the final checkpoints
        DS_JOBS["ds_sync"] = launch(
            "ds_sync",
            "while true; do d=0; "
            f"test -f {LOGS}/ds_gpu1.exit && test -f {LOGS}/ds_gpu2.exit && test -f {LOGS}/ds_gpu3.exit && d=1; "
            f'{sys.executable} -c "{_up}" || true; '
            'test "$d" = 1 && break; sleep 1200; done',
        )
    DS_JOBS
    return DS, DSN


@app.cell
def followups(DS, DSN, LOGS, REPO, job_state, launch, mo, refresh, sys):
    refresh
    # Follow-up studies on the document split: (1) streaming, (2) smaller models, (3) calibration.
    mo.stop(
        not all(job_state(j) == "done" for j in ("ds_gpu1", "ds_gpu3")),
        mo.md("the follow-up studies wait for the document-split training jobs"),
    )
    _res = REPO / "results"
    _runs = REPO / "runs"
    _clean, _ext = DS / "test.jsonl", DS / "external.jsonl"


    def _stream(name, args, sets, extra=()):
        return [
            ["virestore.streaming", *args, "--test", te, *extra, "--tag", f"stream-{name}-{s}", "--results", _res]
            for s, te in sets.items()
        ]


    def _tagger_stream(d):
        return _stream(d, ["--model", "tagger", "--tagger", _runs / f"ds-{d}", "--batch-size", 512], {"clean": _clean, "external": _ext})


    def _quality(d):
        return [
            ["virestore.eval", "--model", "tagger", "--tagger", _runs / f"ds-{d}", "--test", te, "--batch-size", 256,
             "--latency-n", 0, "--nbest", 5, "--tag", f"ds-{d}-{s}", "--results", _res]
            for s, te in {"clean": _clean, "hard": DS / "test_hard.jsonl", "noisy": DSN / "test.jsonl", "external": _ext}.items()
        ]


    def _calib(d):
        return ["virestore.calibration", "--tagger", _runs / f"ds-{d}", "--dev", DS / "dev.jsonl", "--test", _ext,
                "--tag", f"calib-{d}-external", "--results", _res]


    FU_JOBS = {
        "fu_stream": launch(
            "fu_stream",
            *_tagger_stream("tagger"),
            *_tagger_stream("tagger-large"),
            *_stream("lexicon", ["--model", "lexicon", "--train", DS / "train.jsonl"], {"clean": _clean, "external": _ext}),
            ["virestore.tagger", "--config", REPO / "configs/tagger_prefix_ds.yaml"],
            *_quality("tagger-prefix"),
            *_tagger_stream("tagger-prefix"),
        ),
        "fu_small": launch(
            "fu_small",
            ["virestore.tagger", "--config", REPO / "configs/tagger_kd_ds.yaml"],
            *_quality("tagger-kd"),
            ["virestore.tagger", "--config", REPO / "configs/tagger_s_ds.yaml"],
            *_quality("tagger-s"),
            ["virestore.tagger", "--config", REPO / "configs/tagger_s_kd_ds.yaml"],
            *_quality("tagger-s-kd"),
            _calib("tagger"),
            _calib("tagger-large"),
            _calib("tagger-kd"),
        ),
    }
    _up2 = "from huggingface_hub import HfApi; import os; a = HfApi(); " + "; ".join(
        f"os.path.isdir('runs/{d}') and a.upload_folder(repo_id='dinhgia2106/vi-restore-llm-docsplit', folder_path='runs/{d}', path_in_repo='{d}')"
        for d in ("ds-tagger-prefix", "ds-tagger-kd", "ds-tagger-s", "ds-tagger-s-kd")
    )
    FU_JOBS["fu_sync"] = launch(
        "fu_sync",
        "while true; do d=0; "
        f"test -f {LOGS}/fu_stream.exit && test -f {LOGS}/fu_small.exit && d=1; "
        f'{sys.executable} -c "{_up2}" || true; '
        'test "$d" = 1 && break; sleep 1200; done',
    )
    FU_JOBS
    return


@app.cell
def ds_resume(DS, DSN, LOGS, REPO, job_state, launch, mo, refresh, sys):
    refresh
    # Resume after the sandbox loss of 2026-11-23: the TCN checkpoint on HF may predate the end of its
    # training, so it is retrained; TCN evals, its streaming run and the 1.5B LoRA follow. Result JSONs
    # are uploaded to the private repo as they appear, so another shutdown cannot lose them.
    mo.stop(job_state("ds_data") != "done", mo.md("waiting for the document-split data"))
    _res = REPO / "results"
    _sets = {
        "clean": DS / "test.jsonl", "hard": DS / "test_hard.jsonl", "noisy": DSN / "test.jsonl",
        "external": DS / "external.jsonl", "external-noisy": DSN / "external.jsonl",
    }
    _tcn = ["--model", "tcn", "--tcn-repo", "/marimo/Vietnamese-diacritic-restoration", "--tcn-weights", REPO / "runs/ds-tcn/best_model.pth"]
    _q15 = ["--model", "hf", "--hf-model", "Qwen/Qwen2.5-1.5B-Instruct", "--adapter", REPO / "runs/ds-qwen15-lora-noisy", "--batch-size", 64]


    def _evals(name, model_args):
        return [
            ["virestore.eval", *model_args, "--test", te, "--latency-n", 0, "--tag", f"ds-{name}-{s}", "--results", _res, "--save-predictions"]
            for s, te in _sets.items()
        ]


    RESUME_JOBS = {
        "ds_gpu2b": launch(
            "ds_gpu2b",
            "mv runs/ds-tcn runs/ds-tcn-restored",
            ["virestore.train_tcn", "--tcn-repo", "/marimo/Vietnamese-diacritic-restoration", "--train", DS / "train.jsonl",
             "--dev", DS / "dev.jsonl", "--out", REPO / "runs/ds-tcn", "--epochs", 3, "--batch-size", 128, "--limit", 300000],
            *_evals("tcn", _tcn),
            ["virestore.streaming", *_tcn, "--test", DS / "test.jsonl", "--limit", 1000, "--tag", "stream-tcn-clean", "--results", _res],
            ["virestore.train_lora", "--config", REPO / "configs/qwen15_lora_noisy_ds.yaml"],
            *_evals("qwen15-lora-noisy", _q15),
        )
    }
    _up = (
        "from huggingface_hub import HfApi; a = HfApi(); r = 'dinhgia2106/vi-restore-llm-docsplit'; "
        "a.upload_folder(repo_id=r, folder_path='results', path_in_repo='results-wip', allow_patterns=['ds-*.json', 'stream-*.json', 'calib-*.json']); "
        "import os; [a.upload_folder(repo_id=r, folder_path='runs/' + d, path_in_repo=d, ignore_patterns=['*/optimizer.pt', '*/scheduler.pt', '*/rng_state*']) "
        "for d in ('ds-tcn', 'ds-qwen15-lora-noisy') if os.path.isdir('runs/' + d)]"
    )
    RESUME_JOBS["ds_sync2"] = launch(
        "ds_sync2",
        "while true; do d=0; "
        f"test -f {LOGS}/ds_gpu2b.exit && test -f {LOGS}/fu_stream.exit && test -f {LOGS}/fu_small.exit && d=1; "
        f'{sys.executable} -c "{_up}" || true; '
        'test "$d" = 1 && break; sleep 600; done',
    )
    # the first attempt stopped at the TCN streaming run (one-character input, fixed in the adapter)
    RESUME_JOBS["ds_gpu2c"] = launch(
        "ds_gpu2c",
        ["virestore.streaming", *_tcn, "--test", DS / "test.jsonl", "--limit", 1000, "--tag", "stream-tcn-clean", "--results", _res],
        ["virestore.train_lora", "--config", REPO / "configs/qwen15_lora_noisy_ds.yaml"],
        *_evals("qwen15-lora-noisy", _q15),
    )
    RESUME_JOBS["ds_sync3"] = launch(
        "ds_sync3",
        "while true; do d=0; "
        f"test -f {LOGS}/ds_gpu2c.exit && test -f {LOGS}/fu_small.exit && d=1; "
        f'{sys.executable} -c "{_up}" || true; '
        'test "$d" = 1 && break; sleep 600; done',
    )
    RESUME_JOBS

    return


if __name__ == "__main__":
    app.run()
