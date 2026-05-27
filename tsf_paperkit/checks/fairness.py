from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from tsf_paperkit.data.csv_loader import prepare_dataloaders
from tsf_paperkit.runner.experiment import RESULT_COLUMNS, load_config


def entry(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


def check_config(config: dict[str, Any]) -> dict[str, Any]:
    checks = []
    train_cfg = config.get("training", {})
    prepared, loaders = prepare_dataloaders(config["data"], int(train_cfg.get("batch_size", 32)), drop_last=bool(train_cfg.get("test_drop_last", False)))
    checks.append(entry("scaler_train_only", "pass" if prepared.scaler.fit_split == "train" else "fail", f"fit_split={prepared.scaler.fit_split}"))
    checks.append(entry("test_drop_last", "warn" if getattr(loaders["test"], "drop_last", False) else "pass", f"drop_last={getattr(loaders['test'], 'drop_last', False)}"))
    overlap = False
    if prepared.train.dataset and prepared.val.dataset and prepared.train.dataset.metadata and prepared.val.dataset.metadata:
        overlap = prepared.train.dataset.metadata[-1].target_end > prepared.val.dataset.metadata[0].input_start
    checks.append(entry("split_window_overlap", "warn" if overlap else "pass", "checked adjacent train/val windows"))
    checks.append(entry("seed_recorded", "pass" if config.get("experiment", {}).get("seed") is not None else "warn", str(config.get("experiment", {}).get("seed"))))
    out_dir = Path(config.get("experiment", {}).get("output_dir", "results"))
    checks.append(entry("resolved_config_saved", "pass" if (out_dir / "resolved_config.yaml").exists() else "warn", str(out_dir / "resolved_config.yaml")))
    result_path = out_dir / "results.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        missing = [col for col in RESULT_COLUMNS if col not in results.columns]
        checks.append(entry("results_required_columns", "fail" if missing else "pass", f"missing={missing}"))
        if {"num_params", "train_time_sec", "inference_time_sec"}.issubset(results.columns):
            valid_params = (results["num_params"] >= 0).all()
            valid_timing = (results[["train_time_sec", "inference_time_sec"]] >= 0).all().all()
            checks.append(entry("results_num_params_recorded", "pass" if valid_params else "fail", "num_params >= 0"))
            checks.append(entry("results_timing_recorded", "pass" if valid_timing else "fail", "train/inference time >= 0"))
    else:
        checks.append(entry("results_csv_present", "warn", str(result_path)))
    status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return {"status": status, "checks": checks}


def doctor_report() -> dict[str, Any]:
    checks = []
    is_linux = platform.system().lower() == "linux"
    checks.append(entry("os_linux", "pass" if is_linux else "fail", platform.system()))
    checks.append(entry("python_version", "pass" if sys.version_info >= (3, 10) else "fail", sys.version.split()[0]))
    checks.append(entry("torch_import", "pass", torch.__version__))
    for p in [Path(".cache/tsf-paperkit/datasets"), Path(".cache/tsf-paperkit/models")]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
            checks.append(entry(f"cache_writable:{p}", "pass", str(p)))
        except OSError as exc:
            checks.append(entry(f"cache_writable:{p}", "fail", str(exc)))
    cuda = torch.cuda.is_available()
    checks.append(entry("cuda_available", "pass" if cuda else "skipped", str(cuda)))
    checks.append(entry("cuda_smoke", "pass" if cuda else "skipped", "run scripts/smoke_cuda.py for GPU smoke" if cuda else "CUDA unavailable; GPU smoke intentionally skipped"))
    status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return {
        "status": status,
        "checks": checks,
        "environment": {"os": platform.system().lower(), "python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": cuda, "device": "cuda" if cuda else "cpu"},
        "cache": {"datasets": ".cache/tsf-paperkit/datasets", "models": ".cache/tsf-paperkit/models"},
    }


def check_config_file(path: str | Path) -> dict[str, Any]:
    return check_config(load_config(path))


def as_text(report: dict[str, Any]) -> str:
    lines = [f"status: {report['status']}"]
    for c in report.get("checks", []):
        lines.append(f"[{c['status']}] {c['name']}: {c['message']}")
    if "environment" in report:
        lines.append("environment: " + json.dumps(report["environment"], sort_keys=True))
    return "\n".join(lines) + "\n"
