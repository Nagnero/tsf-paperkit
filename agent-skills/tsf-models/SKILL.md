---
name: tsf-models
description: Use when the user wants to list, prepare, cache, or add model adapter recipes and assets for tsf-paperkit experiments.
---
# tsf-models

Goal: manage model recipes and assets without committing weights.

Workflow:
1. Inspect `configs/model_registry.yaml`.
2. List recipes with `python -m tsf_paperkit.cli model list`.
3. Prepare a selected model with `python -m tsf_paperkit.cli model prepare --model <name>`.
4. For built-ins (`naive`, `linear`, `dlinear`), report that no download is required.
5. For future adapters, report provider, revision, license/auth note, and cache path or blocked reason.

Rules:
- Never commit weights, checkpoints, or downloaded model artifacts.
- Use `.cache/tsf-paperkit/models` unless config, env, or `--cache-dir` overrides.
- Do not embed credentials or tokens.
