import json
import subprocess
import sys

import pytest
import torch

from tsf_paperkit.models.adapter_contract import require_optional_dependency
from tsf_paperkit.models.adapters import ADAPTER_SPECS
from tsf_paperkit.models.patchtst import PatchTSTForecastModel, PatchTSTNet
from tsf_paperkit.models.registry import build_model, list_models, prepare_model
from tsf_paperkit.runner.experiment import run_experiment


FUTURE_ADAPTERS = {
    "future.timemixer",
    "future.amd",
    "future.timesfm",
    "future.chronos",
    "future.moirai",
    "future.tirex",
}


def test_future_adapter_specs_are_metadata_only():
    assert FUTURE_ADAPTERS == set(ADAPTER_SPECS)
    for key, spec in ADAPTER_SPECS.items():
        data = spec.as_dict()
        assert data["key"] == key
        assert data["input_shape"] == "batch, seq_len, channels"
        assert data["output_shape"] == "batch, pred_len, channels"
        assert "do not commit weights" in data["asset_policy"]


def test_model_registry_exposes_patchtst_and_future_contracts_without_running_them():
    recipes = {item["name"]: item for item in list_models()}
    assert {"naive", "linear", "dlinear", "patchtst"}.issubset(recipes)
    assert recipes["patchtst"]["status"] == "supported"
    assert recipes["patchtst"]["adapter"] == "builtin.patchtst"
    assert recipes["timesfm_future"]["adapter_contract"]["optional_extra"] == "foundation-adapters"

    ready = prepare_model("patchtst")
    assert ready["status"] == "ready"
    assert ready["message"] == "local model requires no external assets"
    assert ready["cache_path"] is None

    blocked = prepare_model("timesfm_future")
    assert blocked["status"] == "blocked"
    assert blocked["optional_extra"] == "foundation-adapters"


def test_patchtst_forward_shape_and_build():
    net = PatchTSTNet(seq_len=12, pred_len=3, channels=2, patch_len=4, stride=2, d_model=16, n_heads=4, n_layers=1)
    out = net(torch.randn(5, 12, 2))
    assert out.shape == (5, 3, 2)

    model = build_model(
        "patchtst",
        seq_len=12,
        pred_len=3,
        channels=1,
        params={"patch_len": 4, "stride": 2, "d_model": 16, "n_heads": 4, "n_layers": 1},
    )
    assert isinstance(model, PatchTSTForecastModel)
    assert model.num_params() > 0


def test_future_adapter_build_has_actionable_error():
    with pytest.raises(NotImplementedError, match="documented but not runnable"):
        build_model("timemixer_future", seq_len=24, pred_len=12, channels=1)


def test_optional_dependency_error_message_is_actionable():
    with pytest.raises(ImportError, match=r"pip install -e '.\[foundation-adapters\]'"):
        require_optional_dependency("definitely_missing_tsf_paperkit_dep", "foundation-adapters", "TimesFM")


def test_model_cli_lists_and_prepares_patchtst_json():
    listed = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "model", "list"], text=True)
    assert "patchtst" in listed
    assert "builtin.patchtst" in listed

    prepared = subprocess.check_output(
        [sys.executable, "-m", "tsf_paperkit.cli", "model", "prepare", "--model", "patchtst"],
        text=True,
    )
    payload = json.loads(prepared)
    assert payload["status"] == "ready"
    assert payload["model"] == "patchtst"


def test_patchtst_toy_experiment_smoke(tmp_path):
    config = {
        "experiment": {"name": "patchtst_toy", "seed": 42, "output_dir": str(tmp_path / "results")},
        "data": {
            "name": "toy_series",
            "path": "examples/toy_series.csv",
            "timestamp_col": "date",
            "target_cols": ["value"],
            "split": {"train": 0.7, "val": 0.1, "test": 0.2},
            "seq_len": 12,
            "pred_len": 3,
            "stride": 1,
            "scale": True,
        },
        "training": {"batch_size": 8, "epochs": 1, "learning_rate": 0.001, "device": "cpu"},
        "models": [
            {
                "name": "patchtst",
                "params": {"patch_len": 4, "stride": 2, "d_model": 16, "n_heads": 4, "n_layers": 1, "dropout": 0.0},
            }
        ],
    }
    result_path = run_experiment(config)
    text = result_path.read_text()
    assert "patchtst" in text
