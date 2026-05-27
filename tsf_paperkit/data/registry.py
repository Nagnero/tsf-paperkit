from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def resolve_data_cache(cli_cache_dir: str | None = None, config_cache_dir: str | None = None) -> Path:
    return Path(cli_cache_dir or config_cache_dir or os.getenv("TSF_PAPERKIT_DATA_CACHE") or ".cache/tsf-paperkit/datasets")


def load_dataset_registry(path: str | Path = "configs/dataset_registry.yaml") -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return list(data.get("datasets", []))


def dataset_recipe(name: str, path: str | Path = "configs/dataset_registry.yaml") -> dict[str, Any]:
    for recipe in load_dataset_registry(path):
        if recipe.get("name") == name:
            return recipe
    raise KeyError(f"Unknown dataset: {name}")


def prepare_dataset(name: str, cache_dir: str | None = None, registry_path: str | Path = "configs/dataset_registry.yaml") -> dict[str, Any]:
    recipe = dataset_recipe(name, registry_path)
    cache = resolve_data_cache(cache_dir)
    if recipe.get("auth_required"):
        return {"status": "blocked", "dataset": name, "message": "dataset requires user-configured access or license acceptance", "metadata": recipe}
    source = Path(str(recipe.get("source_url", "")))
    cache.mkdir(parents=True, exist_ok=True)
    if source.exists():
        target = cache / recipe.get("cache_key", source.name)
        if target.suffix != source.suffix:
            target = target.with_suffix(source.suffix)
        target.write_bytes(source.read_bytes())
        metadata = {**recipe, "prepared_at": datetime.now(timezone.utc).isoformat(), "local_path": str(target)}
        meta_path = target.with_suffix(target.suffix + ".metadata.yaml")
        meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        return {"status": "ready", "dataset": name, "local_path": str(target), "metadata_path": str(meta_path), "metadata": metadata}
    if recipe.get("placeholder", False):
        return {"status": "skipped", "dataset": name, "message": "placeholder recipe documented; network download not exercised in MVP tests", "metadata": recipe}
    return {"status": "blocked", "dataset": name, "message": f"source not found locally: {source}", "metadata": recipe}
