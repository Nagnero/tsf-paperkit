# tsf-paperkit

**Paper-ready experiment harness for time-series forecasting.**

Run baselines, ablations, reproducibility checks, and Markdown/LaTeX result tables from one small Linux-first repo.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-ready-ee4c2c)](#quick-start)
[![Linux MVP](https://img.shields.io/badge/platform-Linux-lightgrey)](#requirements)
[![Agent Skills](https://img.shields.io/badge/agents-Codex%20%2B%20Claude-7c3aed)](#agent-skills)
[![No SOTA Claims](https://img.shields.io/badge/claims-no%20SOTA-success)](#scope-and-non-goals)

`tsf-paperkit` is not another giant forecasting library.

It is the workflow layer around your forecasting experiments:

- load a dataset once
- run comparable baselines with shared settings
- keep train/val/test scaling honest
- track ablations without hand-merging files
- export paper-style tables without copying numbers
- give Codex or Claude reusable TSF skills that call the same local CLI

If you want a model zoo, use a model zoo. If you want a fast, inspectable path from **clone -> config -> results.csv -> table.tex**, start here.

---

## Recommended default flow

```bash
git clone git@github.com:Nagnero/tsf-paperkit.git
cd tsf-paperkit
pip install -e .

python -m tsf_paperkit.cli doctor --format json
python -m tsf_paperkit.cli run --config configs/example.yaml
python -m tsf_paperkit.cli check --config configs/example.yaml
python -m tsf_paperkit.cli table \
  --input results/toy_experiment/results.csv \
  --format markdown \
  --output results/toy_experiment/table.md
```

That path should give you a toy TSF experiment, reproducibility checks, and a GitHub-readable result table.

---

## What tsf-paperkit is for

Use this repo when you are writing or reviewing a forecasting paper and need a small harness for:

- **baseline runs**: Naive, Linear, DLinear-style MVP models
- **shared settings**: one config controls split, horizon, window length, seed, device, and training settings
- **ablation tracking**: run config overrides and combine results
- **fairness checks**: train-only scaler fitting, split/window overlap checks, `drop_last` warnings, recorded seeds
- **paper output**: Markdown and LaTeX tables with best metrics bolded
- **agent-assisted setup**: portable Codex/Claude skills for data, models, runs, checks, and reports

Do not use it as a benchmark leaderboard or SOTA claim generator.

---

## Quick start

### Requirements

- Linux for official v1 support
- Python 3.10+
- PyTorch
- pandas, numpy, scikit-learn, PyYAML, Typer
- pytest and ruff for development checks
- CUDA is optional; if available, the smoke test runs one toy GPU experiment

### A good first experiment

```bash
# 1. Install locally
pip install -e .

# 2. Verify local readiness
python -m tsf_paperkit.cli doctor --format json

# 3. List dataset/model recipes without downloading large artifacts
python -m tsf_paperkit.cli data list --json
python -m tsf_paperkit.cli model list

# 4. Inspect and prepare only the selected dataset
python -m tsf_paperkit.cli data inspect --dataset toy_series --json
python -m tsf_paperkit.cli data prepare --dataset toy_series --json
python -m tsf_paperkit.cli data prepare --dataset ETTh1 --json
python -m tsf_paperkit.cli data prepare --dataset weather --json
python -m tsf_paperkit.cli model prepare --model dlinear

# 5. Run the toy experiment
python -m tsf_paperkit.cli run --config configs/example.yaml

# 6. Check reproducibility/fairness signals
python -m tsf_paperkit.cli check --config configs/example.yaml --format json

# 7. Export tables
python -m tsf_paperkit.cli table \
  --input results/toy_experiment/results.csv \
  --format latex \
  --output results/toy_experiment/table.tex

python -m pytest
```

If CUDA is available:

```bash
python scripts/smoke_cuda.py
```

CUDA smoke is only a readiness check. It is not a performance result.

---

## A simple mental model

```text
researcher / agent intent
          |
          v
+--------------------+      +-------------------------+
| agent-skills/      | ---> | python -m tsf_paperkit  |
| tsf-* workflows    |      | CLI contract            |
+--------------------+      +-----------+-------------+
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
        v                          v                          v
+---------------+          +---------------+          +----------------+
| data          |          | models        |          | runner/checks  |
| CSV -> splits |          | adapters +    |          | metrics, seed, |
| train scaler  |          | asset recipes |          | ablations      |
| windows       |          |               |          | fairness       |
+-------+-------+          +-------+-------+          +--------+-------+
        |                          |                           |
        +--------------------------+---------------------------+
                                   |
                                   v
                         +-------------------+
                         | reporting         |
                         | results.csv ->    |
                         | Markdown / LaTeX  |
                         +-------------------+
```

The package owns deterministic work. Skills only route intent to CLI commands and scripts.

---

## Start here if you are new

1. Read `configs/example.yaml`.
2. Run `python -m tsf_paperkit.cli doctor --format json`.
3. Run `python -m tsf_paperkit.cli run --config configs/example.yaml`.
4. Inspect `results/toy_experiment/results.csv`.
5. Generate `table.md` or `table.tex`.
6. Copy `configs/example.yaml`, change `seq_len`, `pred_len`, targets, models, or seed.
7. Use `configs/ablation_example.yaml` when you need controlled overrides.

---

## Dataset-first workflow

`tsf-paperkit` uses **selected downloads**. A command should never download every dataset in the registry by default.

```bash
# See available recipes and statuses
python -m tsf_paperkit.cli data list --json
python -m tsf_paperkit.cli data list --family ETT --json

# Inspect provenance before preparing anything
python -m tsf_paperkit.cli data inspect --dataset ETTh1 --json

# Prepare exactly one selected supported dataset
python -m tsf_paperkit.cli data prepare --dataset ETTh1 --json

# Run a small ETT smoke config after preparation
python -m tsf_paperkit.cli run --config configs/ett/etth1_smoke.yaml
python -m tsf_paperkit.cli run --config configs/ett/etth1_patchtst_smoke.yaml
```

`data prepare --json` returns `prepared_files.primary_csv`, `metadata_path`, and a `config_patch` that agents can copy into a run config. Supported ETT recipes download a single checksum-pinned CSV from THUML/Hugging Face. Placeholder datasets return a structured skip reason instead of a fake success.

Current dataset registry status:

| Dataset / family | Status | Source type | Smokeable | Notes |
|---|---:|---|---:|---|
| `toy_series` | supported | local | yes | Bundled demo fixture for fast smoke tests. |
| ETT: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2` | supported | `https_file` | yes | Single-file THUML/Hugging Face downloads with pinned SHA-256 checksums; smoke configs are readiness checks, not benchmarks. |
| Weather | supported | `https_file` | yes | Single-file THUML/Hugging Face download with pinned SHA-256 checksum. |
| Electricity/ECL, Traffic | placeholder | `hf_file` candidate | no | Larger common long-term datasets; no benchmark claims or bulk download. |
| Exchange | supported | `https_file` | yes | Small THUML/Hugging Face CSV with pinned SHA-256 checksum. |
| ILI, Solar-Energy, PeMS | placeholder | candidate/manual | no | Documented because TSF papers use them; enable only after source/license review. |
| Monash tourism monthly | placeholder | manual-only | no | Upstream terms must be checked manually. |

Downloaded data belongs in `.cache/tsf-paperkit/datasets/` or your configured cache path. Do not commit cache contents, generated smoke results, model weights, or checkpoints.

Smoke/demo outputs are readiness checks only. They are not benchmark numbers.

---

## Common CLI surfaces

| Surface | Use it for |
|---|---|
| `doctor` | Linux, Python, PyTorch, CUDA, cache readiness |
| `data list --json` | Show dataset recipes, statuses, source, license, and provenance notes |
| `data inspect --dataset <name> --json` | Inspect one normalized dataset recipe before preparing it |
| `data prepare --dataset <name> --json` | Prepare exactly one supported dataset into local cache |
| `model list` | Show built-in and future model adapter recipes |
| `model prepare` | Prepare model assets or confirm built-ins need none |
| `run` | Execute configured TSF baselines and write `results.csv` |
| `ablate` | Run config overrides and combine ablation outputs |
| `check` | Check leakage/reproducibility/fairness signals |
| `table` | Convert result CSVs to Markdown or LaTeX tables |

Examples:

```bash
python -m tsf_paperkit.cli ablate --config configs/ablation_example.yaml
python -m tsf_paperkit.cli table --input results/toy_experiment/results.csv --format markdown --output results/toy_experiment/table.md
```

---

## Agent skills

The repo includes short, portable skills so Codex and Claude can operate the same harness without inventing commands.

The source of truth is:

```text
agent-skills/*/SKILL.md
```

Generated copies live in:

```text
.codex/skills/*/SKILL.md
.claude/skills/*/SKILL.md
```

Keep them synced with:

```bash
python scripts/sync_agent_skills.py
python scripts/sync_agent_skills.py --check
```

| Skill | Use it for |
|---|---|
| `tsf-doctor` | Verify Linux/Python/PyTorch/CUDA/cache/skill readiness |
| `tsf-data` | Select, inspect, and prepare only requested dataset recipes |
| `tsf-models` | Inspect built-ins and future model asset recipes |
| `tsf-run` | Turn experiment intent into config + run/ablate commands |
| `tsf-check` | Summarize leakage, split, seed, timing, and artifact checks |
| `tsf-report` | Generate Markdown/LaTeX result tables and paper snippets |

Skills are instructions only. They should call `tsf_paperkit.cli` or `scripts/*`, not duplicate Python implementations.

---

## Config shape

```yaml
experiment:
  name: toy_experiment
  seed: 42
  output_dir: results/toy_experiment

data:
  path: examples/toy_series.csv
  timestamp_col: date
  target_cols: ["value"]
  split: {train: 0.7, val: 0.1, test: 0.2}
  seq_len: 12
  pred_len: 3
  stride: 1
  scale: true

training:
  batch_size: 16
  epochs: 2
  learning_rate: 0.01
  device: auto

models:
  - name: naive
  - name: linear
  - name: dlinear
    params: {moving_avg_kernel: 5}
```

Result rows are written to `results.csv` with:

```text
run_id, dataset, model, seq_len, pred_len, seed,
mse, mae, rmse, mape, smape,
train_time_sec, inference_time_sec, num_params
```

---

## Dataset and model artifact policy

This repository stores **recipes, not large artifacts**.

- No third-party datasets are redistributed.
- No model weights or checkpoints are committed.
- Dataset cache defaults to `.cache/tsf-paperkit/datasets/`.
- Non-toy datasets must remain `placeholder` until `configs/dataset_sources.yaml` and `docs/dataset-sources.md` record source/license/provenance approval. ETT, Weather, and Exchange are the first approved non-toy recipes.
- Model cache defaults to `.cache/tsf-paperkit/models/`.
- Environment overrides are supported with `TSF_PAPERKIT_DATA_CACHE` and `TSF_PAPERKIT_MODEL_CACHE`.
- Restricted or authenticated sources should fail clearly until the user accepts upstream terms.

The dataset registry includes the bundled `toy_series`, supported ETT/Weather/Exchange recipes, and placeholder recipes for common TSF datasets such as Electricity/ECL, Traffic, ILI, Solar-Energy, PeMS, and Monash candidates. Placeholder recipes are metadata and provenance prompts, not hidden benchmark downloads.

---

## How it differs

| Project | Primary focus | tsf-paperkit focus |
|---|---|---|
| Time-Series-Library | Broad model implementations and benchmark scripts | Small paper workflow: configs, checks, ablations, tables, skills |
| TFB | Forecasting benchmark evaluation | Clone-to-first-run harness, not a leaderboard |
| BasicTS | Time-series modeling framework | Smaller research-paper experiment kit |
| NeuralForecast | Production-grade neural forecasting models | Minimal baselines plus reproducibility/reporting glue |
| Darts | General time-series toolkit | Paper-ready experiment management and Markdown/LaTeX output |

`tsf-paperkit` complements these projects. It does not replace them.

---

## Adding a model

Runnable MVP models remain deliberately small: `naive`, `linear`, `dlinear`, and `patchtst`. Advanced models are tracked as **future adapters** until their dependency, license, asset, and smoke-test story is clear.

Adapter contract:

```text
input:  batch, seq_len, channels
output: batch, pred_len, channels
```

Promotion flow:

1. Read `docs/model-adapters.md`.
2. Keep new dependencies in optional extras such as `.[research-adapters]` or `.[foundation-adapters]`. The local PatchTST MVP uses only core PyTorch.
3. Implement `BaseForecastModel` only after source/revision/license/weight policy is recorded.
4. Add the adapter key to `MODEL_CLASSES` only when it is runnable.
5. Add metadata to `configs/model_registry.yaml`.
6. Add a tiny CPU smoke test and a config before claiming support.
7. Never commit weights, checkpoints, generated outputs, or downloaded model artifacts.

PatchTST is the first runnable advanced-style adapter. Future adapter placeholders remain recorded for TimeMixer, AMD-style decomposition, TimesFM, Chronos, Moirai, and TiRex. `model prepare` reports those as planned/blocked instead of pretending they are runnable.

---

## Verification checklist

Before publishing changes:

```bash
python scripts/sync_agent_skills.py --check
python -m ruff check .
python -m pytest
python -m tsf_paperkit.cli doctor --format json
python -m tsf_paperkit.cli run --config configs/example.yaml
python -m tsf_paperkit.cli check --config configs/example.yaml --format json
python -m tsf_paperkit.cli table --input results/toy_experiment/results.csv --format markdown --output results/toy_experiment/table.md
python -m tsf_paperkit.cli ablate --config configs/ablation_example.yaml
```

Then confirm Git tracks no generated datasets, model weights, checkpoints, caches, or result artifacts.

---

## Roadmap

- Add more public dataset recipes with license and citation metadata.
- Add real external dataset downloaders after recipe policy is stable.
- Add future model adapters after the built-in adapter contract is proven.
- Add CI after the local Linux MVP path remains stable.
- Add richer paper snippets and appendix checklists.

---

## Scope and non-goals

- Toy/demo outputs are not benchmark results.
- This project does not claim SOTA.
- v1 official support is Linux only.
- PatchTST is implemented as a local MVP adapter; TimeMixer, AMD-style models, TimesFM, Chronos, Moirai, and TiRex remain documented future adapters.
- Real public datasets should be downloaded only through explicit recipes with upstream license/citation awareness.
