# tsf-paperkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-ready-ee4c2c)](#quickstart)
[![Linux MVP](https://img.shields.io/badge/platform-Linux-lightgrey)](#environment)
[![No SOTA claims](https://img.shields.io/badge/claims-no%20SOTA-success)](#scope)

**Paper-ready experiment harness for time-series forecasting: baselines, ablations, fairness checks, and LaTeX tables in one command.**

Many time-series repositories focus on model implementation. **tsf-paperkit focuses on paper-ready experiment management: reproducible runs, ablation tracking, fairness checks, and automatic result tables.**

## Why this exists

TSF researchers often need to answer practical paper questions quickly:

- Can I run the same setup across a few baselines?
- Did I accidentally leak scaler information from test data?
- Can I track ablations without hand-merging CSV files?
- Can I turn results into Markdown/LaTeX tables without copying numbers?
- Can an agent help me prepare data, models, checks, and reports from a cloned repo?

`tsf-paperkit` is not a model zoo and does not claim SOTA. It is a small, extensible harness for fast, inspectable experiments.

## Architecture

```text
User / Agent intent
      |
      v
+-------------------+        +-----------------------+
| agent-skills/     | -----> | tsf_paperkit.cli      |
| tsf-data, tsf-run |        | run/check/table/...   |
+-------------------+        +-----------+-----------+
                                      |
        +-----------------------------+-----------------------------+
        v                             v                             v
+---------------+             +---------------+             +---------------+
| data pipeline |             | model adapters|             | runner/checks |
| CSV, split,   |             | naive, linear,|             | metrics,      |
| train scaler, |             | DLinear,      |             | ablations,    |
| windows       |             | future recipes|             | fairness      |
+---------------+             +---------------+             +---------------+
                                      |
                                      v
                              +---------------+
                              | reporting     |
                              | CSV -> MD/TeX |
                              +---------------+
```

## Installation

```bash
git clone <your-repo-url> tsf-paperkit
cd tsf-paperkit
pip install -e .
```

## Quickstart

```bash
python scripts/sync_agent_skills.py
python -m tsf_paperkit.cli doctor --format json
python -m tsf_paperkit.cli run --config configs/example.yaml
python -m tsf_paperkit.cli check --config configs/example.yaml
python -m tsf_paperkit.cli table \
  --input results/toy_experiment/results.csv \
  --format markdown \
  --output results/toy_experiment/table.md
python -m pytest
```

## Example commands

```bash
# List recipes without downloading everything
python -m tsf_paperkit.cli data list
python -m tsf_paperkit.cli model list

# Prepare local toy dataset / built-in model
python -m tsf_paperkit.cli data prepare --dataset toy_series
python -m tsf_paperkit.cli model prepare --model dlinear

# Run ablations
python -m tsf_paperkit.cli ablate --config configs/ablation_example.yaml

# Generate LaTeX
python -m tsf_paperkit.cli table --input results/toy_experiment/results.csv --format latex --output results/toy_experiment/table.tex
```

## Agent skills

The source of truth is `agent-skills/`. Run `python scripts/sync_agent_skills.py` to copy skills into both `.codex/skills/` and `.claude/skills/`.

| Skill | Role |
|---|---|
| `tsf-doctor` | Linux/Python/PyTorch/CUDA/cache/skill readiness checks |
| `tsf-data` | Dataset recipe listing, preparation, cache, provenance |
| `tsf-models` | Built-in model readiness and future adapter asset recipes |
| `tsf-run` | Config generation/validation, run, ablate |
| `tsf-check` | Reproducibility/fairness/leakage checks |
| `tsf-report` | Results CSV to Markdown/LaTeX and paper snippets |

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

## Dataset and model artifact policy

This repository stores recipes, not large artifacts.

- No third-party datasets are redistributed.
- No model weights/checkpoints are committed.
- Dataset cache defaults to `.cache/tsf-paperkit/datasets/`.
- Model cache defaults to `.cache/tsf-paperkit/models/`.
- Restricted or authenticated sources fail with clear instructions.

## How it differs

| Project | Primary focus | tsf-paperkit difference |
|---|---|---|
| Time-Series-Library | Broad model implementations and benchmark scripts | Focuses on small paper workflow: configs, checks, ablations, tables, skills |
| TFB | Forecasting benchmark evaluation | Lightweight clone-to-first-run harness, not a benchmark leaderboard |
| BasicTS | Time-series modeling framework | Smaller research-paper experiment kit with agent skills |
| NeuralForecast | Production-grade neural forecasting models | Minimal baselines + reproducibility/reporting glue |
| Darts | General time-series toolkit | Paper-ready experiment management and LaTeX/Markdown output |

## Adding a model

1. Implement `BaseForecastModel` in `tsf_paperkit/models/`.
2. Register it in `tsf_paperkit/models/registry.py`.
3. Add metadata to `configs/model_registry.yaml`.
4. Add a smoke config and test.
5. If it requires assets, use lazy cache logic; do not commit weights.

Future adapter TODOs: PatchTST, TimeMixer, AMD-style adaptive multi-scale decomposition, TimesFM, Chronos, Moirai, TiRex.

## Environment

Official v1 support is Linux only. CPU smoke is always expected. If CUDA is available, `scripts/smoke_cuda.py` runs one toy experiment on GPU. CUDA smoke is not a benchmark.

## Publishing checklist

```bash
git init
git add .
git commit -m "Make TSF paper experiments reproducible and agent-friendly"
git remote -v  # ensure an authenticated GitHub remote already exists or add your own
# git push -u origin main
```

Do not push secrets, datasets, downloaded model files, or benchmark claims.

## Roadmap

- Add more public dataset recipes with license/citation metadata.
- Add real future adapters after the registry/cache path is stable.
- Add CI after the MVP local Linux path is proven.
- Add richer paper snippets and appendix checklists.

## Scope

Toy/demo outputs are not benchmark results. This project does not claim SOTA.
