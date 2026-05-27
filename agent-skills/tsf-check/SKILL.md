---
name: tsf-check
description: Use when the user wants reproducibility, fairness, leakage, metric, split, seed, or CUDA readiness checks for tsf-paperkit results.
---
# tsf-check

Goal: audit experiment integrity before reporting results.

Workflow:
1. Run `python -m tsf_paperkit.cli check --config <config>`.
2. For structured output, run `python -m tsf_paperkit.cli check --config <config> --format json`.
3. Inspect results CSV and resolved config when available.
4. Summarize scaler scope, split/window overlap, test drop_last, seed, params, timing, and CUDA smoke status.

Rules:
- Report pass/warn/fail; do not over-certify scientific fairness.
- Route missing or malformed metrics back to tsf-run or tsf-report as appropriate.
