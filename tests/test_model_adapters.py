import json
import subprocess
import sys
from pathlib import Path

import pytest

from tsf_paperkit.models.adapter_contract import require_optional_dependency
from tsf_paperkit.models.adapters import ADAPTER_SPECS
from tsf_paperkit.models.registry import build_model, list_models, prepare_model


FUTURE_ADAPTERS = {
    "future.patchtst",
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


def test_model_registry_exposes_future_contracts_without_running_them():
    recipes = {item["name"]: item for item in list_models()}
    assert {"naive", "linear", "dlinear"}.issubset(recipes)
    assert recipes["patchtst_future"]["adapter_contract"]["key"] == "future.patchtst"
    assert recipes["timesfm_future"]["adapter_contract"]["optional_extra"] == "foundation-adapters"

    planned = prepare_model("patchtst_future")
    assert planned["status"] == "planned"
    assert planned["adapter"] == "future.patchtst"
    assert planned["optional_extra"] == "patchtst"
    assert not Path(planned["cache_path"]).exists()

    blocked = prepare_model("timesfm_future")
    assert blocked["status"] == "blocked"
    assert blocked["optional_extra"] == "foundation-adapters"


def test_future_adapter_build_has_actionable_error():
    with pytest.raises(NotImplementedError, match="documented but not runnable"):
        build_model("patchtst_future", seq_len=24, pred_len=12, channels=1)


def test_optional_dependency_error_message_is_actionable():
    with pytest.raises(ImportError, match=r"pip install -e '.\[patchtst\]'"):
        require_optional_dependency("definitely_missing_tsf_paperkit_dep", "patchtst", "PatchTST")


def test_model_cli_lists_and_prepares_future_adapter_json():
    listed = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "model", "list"], text=True)
    assert "patchtst_future" in listed
    assert "future.patchtst" in listed

    prepared = subprocess.check_output(
        [sys.executable, "-m", "tsf_paperkit.cli", "model", "prepare", "--model", "patchtst_future"],
        text=True,
    )
    payload = json.loads(prepared)
    assert payload["status"] == "planned"
    assert payload["docs"] == "docs/model-adapters.md#patchtst"
