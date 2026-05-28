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
4. For runnable local models (`naive`, `linear`, `dlinear`, `patchtst`), report that no download is required.
5. For future adapters, report provider, revision, optional extra, docs link, license/auth note, and planned/blocked reason. Do not run them.

Rules:
- Never commit weights, checkpoints, or downloaded model artifacts.
- Use `.cache/tsf-paperkit/models` unless config, env, or `--cache-dir` overrides.
- Do not embed credentials or tokens.
- `patchtst` is runnable as a local MVP adapter. Other future adapters are metadata placeholders until their registry entry is promoted and `MODEL_CLASSES` contains a runnable adapter key.
- Use `docs/model-adapters.md` as the adapter contract before adding TimeMixer, AMD-style, TimesFM, Chronos, Moirai, or TiRex support, or before expanding PatchTST beyond the MVP implementation.
