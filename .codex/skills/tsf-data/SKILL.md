---
name: tsf-data
description: Use when the user wants to find, list, download, cache, verify, or configure public time-series datasets for tsf-paperkit.
---
# tsf-data

Goal: help users choose and prepare only the dataset recipes they need, while keeping third-party data out of git.

Workflow:
1. Inspect available recipes with `python -m tsf_paperkit.cli data list --json`.
2. Narrow by intent when useful, e.g. `python -m tsf_paperkit.cli data list --family ETT --json`.
3. Inspect a candidate with `python -m tsf_paperkit.cli data inspect --dataset <name> --json`.
4. Prepare only the selected recipe with `python -m tsf_paperkit.cli data prepare --dataset <name> --json`; ETT recipes are supported, checksum-pinned public CSV downloads.
5. Report `prepared_files.primary_csv`, `metadata_path`, `config_patch`, source, license/citation notes, and cache path.
6. If the recipe is placeholder/restricted/blocked, report the skip or blocked reason instead of inventing a download path.
7. Run a smoke experiment only when the recipe is `smokeable: true` and the user asked to run one. For ETT, prefer `configs/ett/<dataset>_smoke.yaml` and state that output is not a benchmark.

Rules:
- Never download the whole registry by default.
- Never commit downloaded datasets, caches, generated results, checkpoints, weights, or secrets.
- Use `.cache/tsf-paperkit/datasets` unless config, env, or `--cache-dir` overrides.
- Do not bypass source-ledger, license, authentication, or placeholder gates.
- Skills should call CLI commands; do not duplicate deterministic downloader logic in the skill text.
