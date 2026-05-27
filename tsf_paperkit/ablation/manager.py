from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tsf_paperkit.runner.experiment import load_config, run_experiment


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_ablations(path: str | Path) -> Path:
    spec = yaml.safe_load(Path(path).read_text())
    base_config = load_config(spec["base"]["config"])
    root_output = Path(base_config.get("experiment", {}).get("output_dir", "results/ablations")).parent / "ablations"
    root_output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    base_result = run_experiment(base_config)
    base_rows = list(csv.DictReader(base_result.open()))
    for row in base_rows:
        row["ablation"] = "base"
        row["delta_mse_vs_base"] = "0.0"
        all_rows.append(row)
    base_mse_by_model = {r["model"]: float(r["mse"]) for r in base_rows}
    for ab in spec.get("ablations", []):
        cfg = deep_update(deepcopy(base_config), ab.get("override", {}))
        cfg.setdefault("experiment", {})["output_dir"] = str(root_output / ab["name"])
        result = run_experiment(cfg)
        for row in csv.DictReader(result.open()):
            row["ablation"] = ab["name"]
            row["delta_mse_vs_base"] = str(float(row["mse"]) - base_mse_by_model.get(row["model"], float(row["mse"])))
            all_rows.append(row)
    out = root_output / "combined_results.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return out
