import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import yaml

from tsf_paperkit.data.registry import (
    dataset_recipe,
    prepare_dataset,
    resolve_data_cache,
    validate_dataset_registry,
)
from tsf_paperkit.models.assets import resolve_model_cache
from tsf_paperkit.models.registry import prepare_model
from tsf_paperkit.runner.experiment import run_experiment
from tsf_paperkit.runner.trainer import resolve_device


def test_prepare_dataset_and_model_builtin(tmp_path):
    data = prepare_dataset("toy_series", cache_dir=str(tmp_path))
    assert data["status"] == "ready"
    assert Path(data["local_path"]).exists()
    assert Path(data["prepared_files"]["primary_csv"]).exists()
    assert Path(data["metadata_path"]).exists()
    assert data["config_patch"]["data"]["path"] == data["prepared_files"]["primary_csv"]
    model = prepare_model("dlinear")
    assert model["status"] == "ready"
    assert model["cache_path"] is None


def test_dataset_registry_schema_and_aliases():
    recipe = dataset_recipe("toy")
    assert recipe["name"] == "toy_series"
    assert recipe["status"] == "supported"
    ett = dataset_recipe("ETTh1")
    assert ett["status"] == "supported"
    assert ett["source_type"] == "https_file"
    assert ett["source_url"].startswith("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/")
    assert len(ett["sha256"]) == 64
    errors = validate_dataset_registry([
        {
            "name": "bad_supported",
            "status": "supported",
            "source_type": "https_file",
            "source_url": "http://example.com/data.csv",
            "format": "csv",
            "license_note": "review",
            "citation_note": "review",
        }
    ])
    assert any("explicit source_status: approved" in error for error in errors)
    assert any("source_approval_id" in error for error in errors)


def test_registry_validation_rejects_inferred_status_and_alias_collisions():
    missing = validate_dataset_registry([{"name": "implicit", "format": "csv", "license_note": "x", "citation_note": "x"}])
    assert any("missing required fields" in error and "status" in error for error in missing)
    colliding = validate_dataset_registry([
        {
            "name": "first",
            "aliases": ["Shared"],
            "status": "placeholder",
            "source_type": "placeholder",
            "format": "csv",
            "license_note": "x",
            "citation_note": "x",
            "skip_reason": "test",
        },
        {
            "name": "shared",
            "status": "placeholder",
            "source_type": "placeholder",
            "format": "csv",
            "license_note": "x",
            "citation_note": "x",
            "skip_reason": "test",
        },
    ])
    assert any("collides" in error for error in colliding)


def test_custom_registry_requires_sibling_source_ledger(tmp_path):
    registry = tmp_path / "custom_registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "toy_series",
                        "family": "toy",
                        "status": "supported",
                        "source_status": "approved",
                        "source_approval_id": "toy_series_local",
                        "source_type": "local",
                        "source_url": "examples/toy_series.csv",
                        "format": "csv",
                        "license_note": "bundled demo data",
                        "citation_note": "not applicable",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="source_approval_id"):
        dataset_recipe("toy_series", registry)


def test_source_approval_must_be_approved_and_match_recipe():
    recipe = {
        "name": "fixture",
        "family": "fixture",
        "status": "supported",
        "source_status": "approved",
        "source_approval_id": "bad",
        "source_type": "local",
        "source_url": "source.csv",
        "format": "csv",
        "license_note": "x",
        "citation_note": "x",
    }
    errors = validate_dataset_registry(
        [recipe],
        source_approvals={
            "bad": {
                "id": "bad",
                "dataset": "fixture",
                "source_status": "pending",
                "source_type": "https_file",
                "source_url": "other.csv",
            }
        },
    )
    assert any("must have source_status: approved" in error for error in errors)
    assert any("source_type does not match" in error for error in errors)
    assert any("source_url does not match" in error for error in errors)


def test_remote_source_approval_requires_matching_checksum():
    recipe = {
        "name": "remote_fixture",
        "family": "fixture",
        "status": "supported",
        "source_status": "approved",
        "source_approval_id": "remote_fixture",
        "source_type": "https_file",
        "source_url": "https://example.com/source.csv",
        "source_ref": "example/source.csv",
        "format": "csv",
        "license_note": "x",
        "citation_note": "x",
        "sha256": "1" * 64,
    }
    errors = validate_dataset_registry(
        [recipe],
        source_approvals={
            "remote_fixture": {
                "id": "remote_fixture",
                "dataset": "remote_fixture",
                "source_status": "approved",
                "source_type": "https_file",
                "source_url": "https://example.com/source.csv",
                "source_ref": "example/source.csv",
                "sha256": "2" * 64,
            }
        },
    )
    assert any("sha256 does not match" in error for error in errors)

    missing_ledger_checksum = validate_dataset_registry(
        [recipe],
        source_approvals={
            "remote_fixture": {
                "id": "remote_fixture",
                "dataset": "remote_fixture",
                "source_status": "approved",
                "source_type": "https_file",
                "source_url": "https://example.com/source.csv",
                "source_ref": "example/source.csv",
            }
        },
    )
    assert any("source ledger entries require sha256" in error for error in missing_ledger_checksum)


def test_data_list_has_placeholder_metadata():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "data", "list"], text=True)
    assert "toy_series" in out
    assert "ETTh1" in out
    assert "monash_tourism_monthly_placeholder" in out
    assert "license_note" in out
    assert "source_status" in out


def test_data_list_json_and_filters():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "data", "list", "--json", "--family", "ETT"], text=True)
    data = json.loads(out)
    names = {item["name"] for item in data["datasets"]}
    assert {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}.issubset(names)
    assert "toy_series" not in names


def test_data_inspect_json_alias():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "data", "inspect", "--dataset", "toy", "--json"], text=True)
    data = json.loads(out)
    assert data["dataset"]["name"] == "toy_series"


def test_data_prepare_json_contract(tmp_path):
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "tsf_paperkit.cli",
            "data",
            "prepare",
            "--dataset",
            "toy_series",
            "--json",
            "--cache-dir",
            str(tmp_path),
        ],
        text=True,
    )
    data = json.loads(out)
    assert data["status"] == "ready"
    assert Path(data["prepared_files"]["primary_csv"]).exists()
    assert Path(data["metadata_path"]).exists()
    assert data["config_patch"]["data"]["path"] == data["prepared_files"]["primary_csv"]


def test_placeholder_prepare_skips_without_cache_write(tmp_path):
    data = prepare_dataset("electricity", cache_dir=str(tmp_path))
    assert data["status"] == "skipped"
    assert "source ledger pending" in data["message"]
    assert not any(tmp_path.rglob("*.csv"))


def test_bare_prepare_is_rejected(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "tsf_paperkit.cli", "data", "prepare", "--cache-dir", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert not any(tmp_path.iterdir())


def test_family_prepare_previews_and_does_not_prepare_placeholders(tmp_path):
    out = subprocess.check_output(
        [sys.executable, "-m", "tsf_paperkit.cli", "data", "prepare", "--family", "ETT", "--json", "--cache-dir", str(tmp_path)],
        text=True,
    )
    data = json.loads(out)
    assert data["status"] == "preview"
    assert data["preview"] == ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    assert data["prepared"] == []
    assert data["requires_confirmation"] is True
    assert data["skipped"] == []
    assert not any(tmp_path.rglob("*.csv"))


def test_unknown_family_prepare_is_rejected(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsf_paperkit.cli",
            "data",
            "prepare",
            "--family",
            "does-not-exist",
            "--cache-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert not any(tmp_path.iterdir())


def test_local_and_zip_fixture_prepare(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("date,value\n2024-01-01,1\n2024-01-02,2\n")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source, "inside/source.csv")
    registry = tmp_path / "registry.yaml"
    (tmp_path / "dataset_sources.yaml").write_text(
        yaml.safe_dump(
            {
                "approved_sources": [
                    {"id": "fixture_local", "dataset": "fixture_local", "source_status": "approved", "source_type": "local", "source_url": str(source)},
                    {"id": "fixture_zip", "dataset": "fixture_zip", "source_status": "approved", "source_type": "zip_archive", "source_url": str(archive)},
                    {"id": "fixture_bad_checksum", "dataset": "fixture_bad_checksum", "source_status": "approved", "source_type": "local", "source_url": str(source)},
                ]
            }
        )
    )
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "fixture_local",
                        "family": "fixture",
                        "status": "supported",
                        "source_status": "approved",
                        "source_approval_id": "fixture_local",
                        "source_type": "local",
                        "source_url": str(source),
                        "format": "csv",
                        "license_note": "test fixture",
                        "citation_note": "not applicable",
                        "timestamp_col": "date",
                        "target_cols": ["value"],
                        "cache_key": "fixture_local",
                        "cache_subdir": "fixture_local",
                        "prepared_files": {"primary_csv": "source.csv"},
                    },
                    {
                        "name": "fixture_zip",
                        "family": "fixture",
                        "status": "supported",
                        "source_status": "approved",
                        "source_approval_id": "fixture_zip",
                        "source_type": "zip_archive",
                        "source_url": str(archive),
                        "archive_member": "inside/source.csv",
                        "format": "zip_csv",
                        "license_note": "test fixture",
                        "citation_note": "not applicable",
                        "timestamp_col": "date",
                        "target_cols": ["value"],
                        "cache_key": "fixture_zip",
                        "cache_subdir": "fixture_zip",
                        "prepared_files": {"primary_csv": "source.csv"},
                    },
                    {
                        "name": "fixture_bad_checksum",
                        "family": "fixture",
                        "status": "supported",
                        "source_status": "approved",
                        "source_approval_id": "fixture_bad_checksum",
                        "source_type": "local",
                        "source_url": str(source),
                        "format": "csv",
                        "license_note": "test fixture",
                        "citation_note": "not applicable",
                        "timestamp_col": "date",
                        "target_cols": ["value"],
                        "cache_key": "fixture_bad_checksum",
                        "cache_subdir": "fixture_bad_checksum",
                        "prepared_files": {"primary_csv": "source.csv"},
                        "sha256": "0" * 64,
                    },
                ]
            },
            sort_keys=False,
        )
    )
    local = prepare_dataset("fixture_local", cache_dir=str(tmp_path / "cache"), registry_path=registry)
    zipped = prepare_dataset("fixture_zip", cache_dir=str(tmp_path / "cache"), registry_path=registry)
    bad_checksum = prepare_dataset("fixture_bad_checksum", cache_dir=str(tmp_path / "cache"), registry_path=registry)
    assert local["status"] == "ready"
    assert zipped["status"] == "ready"
    assert bad_checksum["status"] == "blocked"
    assert not (tmp_path / "cache" / "fixture_bad_checksum" / "source.csv").exists()
    assert Path(local["prepared_files"]["primary_csv"]).read_text().startswith("date,value")
    assert Path(zipped["prepared_files"]["primary_csv"]).read_text().startswith("date,value")


def test_prepare_output_can_feed_experiment_runner(tmp_path):
    prepared = prepare_dataset("toy_series", cache_dir=str(tmp_path / "cache"))
    data_cfg = prepared["config_patch"]["data"]
    config = {
        "experiment": {"name": "prepared_fixture", "seed": 7, "output_dir": str(tmp_path / "results")},
        "data": {
            **data_cfg,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "seq_len": 4,
            "pred_len": 2,
            "stride": 1,
            "scale": True,
        },
        "training": {"batch_size": 4, "epochs": 1, "learning_rate": 0.001, "device": "cpu"},
        "models": [{"name": "naive"}],
    }
    result_path = run_experiment(config)
    assert result_path.exists()
    assert "naive" in result_path.read_text()


def test_cache_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TSF_PAPERKIT_DATA_CACHE", str(tmp_path / "env"))
    assert resolve_data_cache() == tmp_path / "env"
    assert resolve_data_cache(config_cache_dir=str(tmp_path / "cfg")) == tmp_path / "cfg"
    assert resolve_data_cache(cli_cache_dir=str(tmp_path / "cli")) == tmp_path / "cli"
    monkeypatch.setenv("TSF_PAPERKIT_MODEL_CACHE", str(tmp_path / "model-env"))
    assert resolve_model_cache() == tmp_path / "model-env"
    assert resolve_model_cache(config_cache_dir=str(tmp_path / "model-cfg")) == tmp_path / "model-cfg"
    assert resolve_model_cache(cli_cache_dir=str(tmp_path / "model-cli")) == tmp_path / "model-cli"


def test_import_does_not_download_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.check_call([sys.executable, "-c", "import tsf_paperkit; import tsf_paperkit.models.registry; import tsf_paperkit.data.registry"])
    assert not (tmp_path / ".cache").exists()


def test_device_resolution_cpu_and_cuda_failure():
    assert resolve_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")


def test_skill_sync_and_check():
    subprocess.run([sys.executable, "scripts/sync_agent_skills.py", "--check"], check=True)
    for skill in ["tsf-doctor", "tsf-data", "tsf-models", "tsf-run", "tsf-check", "tsf-report"]:
        src = Path("agent-skills") / skill / "SKILL.md"
        assert src.read_bytes() == (Path(".codex/skills") / skill / "SKILL.md").read_bytes()
        assert src.read_bytes() == (Path(".claude/skills") / skill / "SKILL.md").read_bytes()


def test_doctor_json_contract():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "doctor", "--format", "json"], text=True)
    data = json.loads(out)
    assert {"status", "checks", "environment", "cache"}.issubset(data)
