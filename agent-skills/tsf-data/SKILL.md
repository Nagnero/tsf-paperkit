---
name: tsf-data
description: Use when the user wants to find, list, download, cache, verify, or configure public time-series datasets for tsf-paperkit.
---
# tsf-data

Goal: prepare datasets from registry recipes without committing data to git.

Workflow:
1. Inspect `configs/dataset_registry.yaml`.
2. List recipes with `python -m tsf_paperkit.cli data list`.
3. Prepare a selected recipe with `python -m tsf_paperkit.cli data prepare --dataset <name>`.
4. Report local path, metadata path, source URL, license note, checksum, and auth status.
5. If a source is placeholder/restricted, explain the skip or required user access.

Rules:
- Never commit downloaded datasets.
- Use `.cache/tsf-paperkit/datasets` unless config, env, or `--cache-dir` overrides.
- Do not bypass license or authentication gates.
