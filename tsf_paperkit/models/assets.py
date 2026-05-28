from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_model_cache(cli_cache_dir: str | None = None, config_cache_dir: str | None = None) -> Path:
    return Path(cli_cache_dir or config_cache_dir or os.getenv("TSF_PAPERKIT_MODEL_CACHE") or ".cache/tsf-paperkit/models")


def prepare_model_asset(recipe: dict[str, Any], cache_dir: str | None = None, config_cache_dir: str | None = None) -> dict[str, Any]:
    cache = resolve_model_cache(cache_dir, config_cache_dir)
    if recipe.get("kind") == "builtin":
        return {"status": "ready", "model": recipe["name"], "cache_path": None, "message": "built-in model requires no external assets"}
    if recipe.get("kind") == "future-adapter":
        status = "blocked" if recipe.get("auth_required") else "planned"
        return {
            "status": status,
            "model": recipe["name"],
            "cache_path": str(cache / recipe.get("cache_key", recipe["name"])),
            "message": "future adapter is documented but not implemented; no assets were downloaded",
            "adapter": recipe.get("adapter"),
            "optional_extra": recipe.get("optional_extra"),
            "docs": recipe.get("docs"),
            "license_note": recipe.get("license_note"),
        }
    if recipe.get("auth_required"):
        return {"status": "blocked", "model": recipe["name"], "cache_path": str(cache), "message": "model requires user-configured credentials or license acceptance"}
    cache.mkdir(parents=True, exist_ok=True)
    return {"status": "ready", "model": recipe["name"], "cache_path": str(cache / recipe.get("cache_key", recipe["name"])), "message": "external adapter recipe recorded; downloader not implemented in MVP"}
