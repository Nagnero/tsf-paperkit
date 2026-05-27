from __future__ import annotations

import csv
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tsf_paperkit.data.csv_loader import prepare_dataloaders
from tsf_paperkit.models.registry import build_model, prepare_model
from tsf_paperkit.runner.evaluator import evaluate_predictions
from tsf_paperkit.runner.trainer import count_params, resolve_device, set_seed

RESULT_COLUMNS = ["run_id", "dataset", "model", "seq_len", "pred_len", "seed", "mse", "mae", "rmse", "mape", "smape", "train_time_sec", "inference_time_sec", "num_params"]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def save_resolved_config(config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_experiment(config: dict[str, Any]) -> Path:
    config = deepcopy(config)
    exp = config.get("experiment", {})
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    seed = int(exp.get("seed", train_cfg.get("seed", 42)))
    set_seed(seed)
    device = resolve_device(train_cfg.get("device", "auto"))
    output_dir = Path(exp.get("output_dir", "results/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("runtime", {})["resolved_device"] = device
    save_resolved_config(config, output_dir)

    prepared, loaders = prepare_dataloaders(data_cfg, batch_size=int(train_cfg.get("batch_size", 32)), drop_last=bool(train_cfg.get("test_drop_last", False)))
    channels = len(prepared.target_cols)
    rows = []
    for model_cfg in config.get("models", []):
        name = model_cfg["name"]
        prepare_model(name)
        model = build_model(name, int(data_cfg["seq_len"]), int(data_cfg["pred_len"]), channels, device=device, params=model_cfg.get("params", {}))
        t0 = time.perf_counter()
        model.fit(loaders["train"], loaders.get("val"), {**train_cfg, **model_cfg.get("params", {})})
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred, y_true = model.predict(loaders["test"])
        infer_time = time.perf_counter() - t1
        metrics = evaluate_predictions(y_true, y_pred)
        rows.append({
            "run_id": str(uuid.uuid4())[:8],
            "dataset": data_cfg.get("name") or Path(data_cfg["path"]).stem,
            "model": name,
            "seq_len": int(data_cfg["seq_len"]),
            "pred_len": int(data_cfg["pred_len"]),
            "seed": seed,
            **metrics,
            "train_time_sec": round(train_time, 6),
            "inference_time_sec": round(infer_time, 6),
            "num_params": count_params(model),
        })
    result_path = output_dir / "results.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return result_path
