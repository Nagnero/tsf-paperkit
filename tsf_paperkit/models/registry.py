from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tsf_paperkit.models.assets import prepare_model_asset
from tsf_paperkit.models.dlinear import DLinearForecastModel
from tsf_paperkit.models.linear import LinearForecastModel
from tsf_paperkit.models.naive import NaiveLastValue

MODEL_CLASSES = {
    "builtin.naive": NaiveLastValue,
    "builtin.linear": LinearForecastModel,
    "builtin.dlinear": DLinearForecastModel,
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
    legacy_adapter = f"builtin.{name}"
    if legacy_adapter in MODEL_CLASSES:
        return {"name": name, "kind": "builtin", "provider": "tsf-paperkit", "revision": "local", "expected_files": [], "cache_key": None, "license_note": "project license", "auth_required": False, "adapter": legacy_adapter}
    raise KeyError(f"Unknown model: {name}")


def build_model(name: str, seq_len: int, pred_len: int, channels: int, device: str = "cpu", params: dict[str, Any] | None = None, registry_path: str | Path = "configs/model_registry.yaml"):
    params = params or {}
    recipe = model_recipe(name, registry_path)
    adapter = recipe.get("adapter")
    model_cls = MODEL_CLASSES.get(adapter)
    if model_cls is None:
        raise KeyError(f"Model {name!r} uses adapter {adapter!r}, which is not implemented in the MVP registry.")
    if adapter == "builtin.naive":
        return model_cls()
    return model_cls(seq_len, pred_len, channels, device=device, **params)


def prepare_model(name: str, cache_dir: str | None = None, registry_path: str | Path = "configs/model_registry.yaml", config_cache_dir: str | None = None) -> dict[str, Any]:
    return prepare_model_asset(model_recipe(name, registry_path), cache_dir=cache_dir, config_cache_dir=config_cache_dir)
