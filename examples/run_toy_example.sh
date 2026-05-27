#!/usr/bin/env bash
set -euo pipefail
python -m tsf_paperkit.cli run --config configs/example.yaml
python -m tsf_paperkit.cli check --config configs/example.yaml
python -m tsf_paperkit.cli table --input results/toy_experiment/results.csv --format markdown --output results/toy_experiment/table.md
