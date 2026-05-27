---
name: tsf-report
description: Use when the user wants to convert tsf-paperkit results CSV files into Markdown, LaTeX, README snippets, or paper-ready result summaries.
---
# tsf-report

Goal: produce GitHub/paper-readable outputs from results CSV.

Workflow:
1. Confirm the results CSV exists and includes required metrics.
2. Generate Markdown with `python -m tsf_paperkit.cli table --input <csv> --format markdown --output <path>`.
3. Generate LaTeX with `python -m tsf_paperkit.cli table --input <csv> --format latex --output <path>`.
4. Summarize best scores and include toy/demo caveats when applicable.

Rules:
- Do not invent numbers.
- Do not claim SOTA.
- If metrics are missing, route to `python -m tsf_paperkit.cli check --config <config>`.
