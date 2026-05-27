from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tsf_paperkit.models.assets import prepare_model_asset
from tsf_paperkit.models.dlinear import DLinearForecastModel
from tsf_paperkit.models.linear import LinearForecastModel
from tsf_paperkit.models.naive import NaiveLastValue

MODEL_CLASSES = {
    "naive": NaiveLastValue,
    "linear": LinearForecastModel,
    "dlinear": DLinearForecastModel,
}


def load_model_registry(path: str | Path = "configs/model_registry.yaml") -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return list(data.get("models", []))


def list_models(path: str | Path = "configs/model_registry.yaml") -> list[dict[str, Any]]:
    return load_model_registry(path)


def model_recipe(name: str, path: str | Path = "configs/model_registry.yaml") -> dict[str, Any]:
    for recipe in load_model_registry(path):
        if recipe.get("name") == name:
            return recipe
    if name in MODEL_CLASSES:
        return {"name": name, "kind": "builtin", "provider": "tsf-paperkit", "revision": "local", "expected_files": [], "cache_key": None, "license_note": "project license", "auth_required": False}
    raise KeyError(f"Unknown model: {name}")


def build_model(name: str, seq_len: int, pred_len: int, channels: int, device: str = "cpu", params: dict[str, Any] | None = None):
    params = params or {}
    if name == "naive":
        return NaiveLastValue()
    if name == "linear":
        return LinearForecastModel(seq_len, pred_len, channels, device=device)
    if name == "dlinear":
        return DLinearForecastModel(seq_len, pred_len, channels, device=device, **params)
    raise KeyError(f"Model {name!r} is not implemented in MVP. Add an adapter recipe first.")


def prepare_model(name: str, cache_dir: str | None = None, registry_path: str | Path = "configs/model_registry.yaml") -> dict[str, Any]:
    return prepare_model_asset(model_recipe(name, registry_path), cache_dir=cache_dir)
