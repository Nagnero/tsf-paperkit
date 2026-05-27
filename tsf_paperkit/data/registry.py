from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REGISTRY_STATUSES = {"supported", "placeholder", "blocked", "restricted"}
SOURCE_TYPES = {"local", "https_file", "zip_archive", "hf_file", "manual_only", "placeholder"}
REQUIRED_FIELDS = {"name", "status", "source_type", "license_note", "citation_note", "format"}


def resolve_data_cache(cli_cache_dir: str | None = None, config_cache_dir: str | None = None) -> Path:
    return Path(cli_cache_dir or config_cache_dir or os.getenv("TSF_PAPERKIT_DATA_CACHE") or ".cache/tsf-paperkit/datasets")


def load_dataset_registry(path: str | Path = "configs/dataset_registry.yaml", *, validate: bool = False) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    raw_recipes = list(data.get("datasets", []))
    source_approvals = load_source_approvals(_default_source_ledger_path(p)) if validate else {}
    if validate:
        errors = validate_dataset_registry(raw_recipes, source_approvals=source_approvals)
        if errors:
            raise ValueError("Invalid dataset registry:\n" + "\n".join(f"- {e}" for e in errors))
    return [normalize_recipe(recipe) for recipe in raw_recipes]


def _default_source_ledger_path(registry_path: Path) -> Path:
    sibling = registry_path.with_name("dataset_sources.yaml")
    default_registry = Path("configs/dataset_registry.yaml")
    if registry_path == default_registry or registry_path.resolve() == default_registry.resolve():
        return Path("configs/dataset_sources.yaml")
    return sibling


def load_source_approvals(path: str | Path = "configs/dataset_sources.yaml") -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    approvals = data.get("approved_sources", []) or []
    return {str(item.get("id")): dict(item) for item in approvals if item.get("id")}


def normalize_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(recipe)
    normalized.setdefault("status", "placeholder" if normalized.get("placeholder") else "supported")
    normalized.setdefault("source_type", "placeholder" if normalized.get("placeholder") else "local")
    normalized.setdefault("aliases", [])
    normalized.setdefault("family", normalized.get("name"))
    normalized.setdefault("tags", [])
    normalized.setdefault("expected_files", [])
    normalized.setdefault("prepared_files", {})
    normalized.setdefault("size_note", normalized.get("size_mb"))
    normalized.setdefault("smokeable", normalized.get("status") == "supported")
    normalized.setdefault("source_status", "pending")
    if "target_columns" in normalized and "target_cols" not in normalized:
        normalized["target_cols"] = normalized["target_columns"]
    if "checksum" in normalized and "sha256" not in normalized:
        normalized["sha256"] = normalized["checksum"]
    normalized["aliases"] = list(normalized.get("aliases") or [])
    normalized["target_cols"] = list(normalized.get("target_cols") or [])
    normalized["expected_files"] = list(normalized.get("expected_files") or [])
    return normalized


def validate_dataset_registry(
    recipes: list[dict[str, Any]],
    *,
    source_approvals: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    seen_keys: dict[str, str] = {}
    source_approvals = source_approvals or {}
    for index, raw_recipe in enumerate(recipes):
        recipe = dict(raw_recipe)
        name = recipe.get("name") or f"<entry {index}>"
        missing = sorted(field for field in REQUIRED_FIELDS if recipe.get(field) in (None, ""))
        if missing:
            errors.append(f"{name}: missing required fields {missing}")
        status = recipe.get("status")
        if status not in REGISTRY_STATUSES:
            errors.append(f"{name}: status must be one of {sorted(REGISTRY_STATUSES)}, got {status!r}")
        source_type = recipe.get("source_type")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{name}: source_type must be one of {sorted(SOURCE_TYPES)}, got {source_type!r}")
        for key_kind, key_value in [("name", name), *(("alias", alias) for alias in recipe.get("aliases", []) or [])]:
            lookup = str(key_value).lower()
            if lookup in seen_keys:
                errors.append(f"{name}: {key_kind} {key_value!r} collides with {seen_keys[lookup]}")
            seen_keys[lookup] = f"{key_kind} on {name}"
        if status in {"placeholder", "blocked", "restricted"} and not (recipe.get("skip_reason") or recipe.get("blocked_reason")):
            errors.append(f"{name}: {status} entries require skip_reason or blocked_reason")
        if status == "supported":
            approval_id = recipe.get("source_approval_id")
            approval = source_approvals.get(str(approval_id)) if approval_id else None
            if recipe.get("source_status") != "approved":
                errors.append(f"{name}: supported recipes require explicit source_status: approved")
            if not approval:
                errors.append(f"{name}: supported recipes require source_approval_id matching dataset_sources.yaml")
            else:
                if approval.get("source_status") != "approved":
                    errors.append(f"{name}: source approval {approval_id!r} must have source_status: approved")
                if approval.get("dataset") != name:
                    errors.append(f"{name}: source approval {approval_id!r} is for {approval.get('dataset')!r}")
                if approval.get("source_type") != source_type:
                    errors.append(f"{name}: source approval {approval_id!r} source_type does not match recipe")
                for source_field in ("source_url", "source_ref"):
                    if recipe.get(source_field) and approval.get(source_field) != recipe.get(source_field):
                        errors.append(f"{name}: source approval {approval_id!r} {source_field} does not match recipe")
            source_url = str(recipe.get("source_url") or "")
            is_remote = source_url.startswith(("http://", "https://"))
            if is_remote and not source_url.startswith("https://"):
                errors.append(f"{name}: approved remote sources must use https")
            if is_remote and not (recipe.get("sha256") or recipe.get("checksum") or recipe.get("checksum_waiver")):
                errors.append(f"{name}: approved remote sources require sha256 or checksum_waiver")
    return errors


def dataset_recipe(name: str, path: str | Path = "configs/dataset_registry.yaml") -> dict[str, Any]:
    lookup = name.lower()
    for recipe in load_dataset_registry(path, validate=True):
        keys = [recipe.get("name", ""), *recipe.get("aliases", [])]
        if any(str(key).lower() == lookup for key in keys):
            return recipe
    raise KeyError(f"Unknown dataset: {name}")


def list_dataset_recipes(path: str | Path = "configs/dataset_registry.yaml", family: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    recipes = load_dataset_registry(path, validate=True)
    if family:
        family_lower = family.lower()
        recipes = [r for r in recipes if str(r.get("family", "")).lower() == family_lower]
    if status:
        status_lower = status.lower()
        recipes = [r for r in recipes if str(r.get("status", "")).lower() == status_lower]
    return recipes


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_download(source: str, target: Path) -> None:
    source_path = Path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    if source_path.exists():
        shutil.copyfile(source_path, tmp_target)
        tmp_target.replace(target)
        return
    if source.startswith("file://"):
        shutil.copyfile(Path(source.removeprefix("file://")), tmp_target)
        tmp_target.replace(target)
        return
    if source.startswith("http://"):
        raise ValueError("approved remote downloads must use https")
    if source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=30) as response:  # noqa: S310 - explicit user-selected public recipe download
            with tmp_target.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp_target.replace(target)
        return
    raise FileNotFoundError(f"source not found locally: {source}")


def _target_filename(recipe: dict[str, Any], source: str) -> str:
    prepared = recipe.get("prepared_files") or {}
    if prepared.get("primary_csv"):
        return Path(str(prepared["primary_csv"])).name
    cache_key = recipe.get("cache_key") or recipe["name"]
    suffix = Path(source).suffix or ".csv"
    if str(cache_key).endswith(suffix):
        return str(cache_key)
    return f"{cache_key}{suffix}"


def _config_patch(recipe: dict[str, Any], primary_csv: Path | None) -> dict[str, Any]:
    patch = {
        "data": {
            "name": recipe.get("name"),
            "path": str(primary_csv) if primary_csv else None,
            "timestamp_col": recipe.get("timestamp_col"),
            "target_cols": recipe.get("target_cols", []),
        }
    }
    if recipe.get("freq"):
        patch["data"]["freq"] = recipe["freq"]
    return patch


def _write_metadata(recipe: dict[str, Any], dataset_dir: Path, prepared_files: dict[str, str], status: str, message: str | None = None) -> Path:
    primary = Path(prepared_files["primary_csv"]) if prepared_files.get("primary_csv") else None
    metadata = {
        "dataset": recipe["name"],
        "status": status,
        "message": message,
        "source_type": recipe.get("source_type"),
        "source_url": recipe.get("source_url"),
        "source_ref": recipe.get("source_ref"),
        "source_status": recipe.get("source_status"),
        "license_note": recipe.get("license_note"),
        "citation_note": recipe.get("citation_note"),
        "prepared_at": _now(),
        "cache_path": str(dataset_dir),
        "prepared_files": prepared_files,
        "checksum": {"sha256": _sha256(primary)} if primary and primary.exists() else None,
        "config_patch": _config_patch(recipe, primary),
    }
    meta_path = dataset_dir / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return meta_path


def _blocked(recipe: dict[str, Any], status: str, message: str, cache: Path | None = None) -> dict[str, Any]:
    result = {
        "status": status,
        "dataset": recipe.get("name"),
        "message": message,
        "cache_path": str(cache) if cache else None,
        "prepared_files": {},
        "metadata_path": None,
        "config_patch": _config_patch(recipe, None),
        "metadata": recipe,
    }
    return result


def prepare_dataset(
    name: str,
    cache_dir: str | None = None,
    registry_path: str | Path = "configs/dataset_registry.yaml",
    *,
    force: bool = False,
) -> dict[str, Any]:
    recipe = dataset_recipe(name, registry_path)
    cache = resolve_data_cache(cache_dir)
    dataset_dir = cache / str(recipe.get("cache_subdir") or recipe.get("cache_key") or recipe["name"])

    if recipe.get("auth_required") or recipe.get("status") == "restricted":
        return _blocked(recipe, "blocked", recipe.get("blocked_reason") or "dataset requires user-configured access or license acceptance", dataset_dir)
    if recipe.get("status") in {"placeholder", "blocked"} or recipe.get("source_type") in {"placeholder", "manual_only", "hf_file"}:
        reason = recipe.get("skip_reason") or recipe.get("blocked_reason") or "placeholder recipe documented; automatic download is not enabled"
        return _blocked(recipe, "skipped" if recipe.get("status") == "placeholder" else "blocked", reason, dataset_dir)

    source = str(recipe.get("source_url") or "")
    if not source:
        return _blocked(recipe, "blocked", "recipe has no source_url", dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_type = recipe.get("source_type")
        if source_type in {"local", "https_file"}:
            target = dataset_dir / _target_filename(recipe, source)
            if force or not target.exists():
                _copy_or_download(source, target)
            primary_csv = target
        elif source_type == "zip_archive":
            archive = dataset_dir / _target_filename({**recipe, "prepared_files": {}}, source)
            if force or not archive.exists():
                _copy_or_download(source, archive)
            member = recipe.get("archive_member")
            if not member:
                return _blocked(recipe, "blocked", "zip_archive recipes require archive_member", dataset_dir)
            with zipfile.ZipFile(archive) as zf:
                if member not in zf.namelist():
                    return _blocked(recipe, "blocked", f"archive member not found: {member}", dataset_dir)
                primary_csv = dataset_dir / Path(member).name
                if force or not primary_csv.exists():
                    with zf.open(member) as src, primary_csv.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            return _blocked(recipe, "blocked", f"unsupported source_type: {source_type}", dataset_dir)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as exc:
        return _blocked(recipe, "blocked", str(exc), dataset_dir)

    expected_sha = recipe.get("sha256")
    if expected_sha and _sha256(primary_csv) != expected_sha:
        primary_csv.unlink(missing_ok=True)
        return _blocked(recipe, "blocked", "sha256 checksum mismatch", dataset_dir)

    prepared_files = {"primary_csv": str(primary_csv)}
    meta_path = _write_metadata(recipe, dataset_dir, prepared_files, "ready")
    metadata = yaml.safe_load(meta_path.read_text())
    return {
        "status": "ready",
        "dataset": recipe["name"],
        "local_path": str(primary_csv),
        "cache_path": str(dataset_dir),
        "prepared_files": prepared_files,
        "metadata_path": str(meta_path),
        "config_patch": metadata["config_patch"],
        "metadata": metadata,
    }
