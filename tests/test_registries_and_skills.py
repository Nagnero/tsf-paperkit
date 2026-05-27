import json
import subprocess
import sys
from pathlib import Path

import torch

from tsf_paperkit.data.registry import prepare_dataset, resolve_data_cache
from tsf_paperkit.models.registry import prepare_model
from tsf_paperkit.runner.trainer import resolve_device


def test_prepare_dataset_and_model_builtin():
    data = prepare_dataset("toy_series")
    assert data["status"] == "ready"
    assert Path(data["local_path"]).exists()
    model = prepare_model("dlinear")
    assert model["status"] == "ready"
    assert model["cache_path"] is None


def test_data_list_has_placeholder_metadata():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "data", "list"], text=True)
    assert "toy_series" in out
    assert "monash_tourism_monthly_placeholder" in out
    assert "license_note" in out


def test_cache_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TSF_PAPERKIT_DATA_CACHE", str(tmp_path / "env"))
    assert resolve_data_cache() == tmp_path / "env"
    assert resolve_data_cache(config_cache_dir=str(tmp_path / "cfg")) == tmp_path / "cfg"
    assert resolve_data_cache(cli_cache_dir=str(tmp_path / "cli")) == tmp_path / "cli"


def test_device_resolution_cpu_and_cuda_failure():
    assert resolve_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        try:
            resolve_device("cuda")
        except RuntimeError as exc:
            assert "cuda" in str(exc).lower()
        else:
            raise AssertionError("cuda should fail when unavailable")


def test_skill_sync_and_check():
    subprocess.run([sys.executable, "scripts/sync_agent_skills.py"], check=True)
    subprocess.run([sys.executable, "scripts/sync_agent_skills.py", "--check"], check=True)
    for skill in ["tsf-doctor", "tsf-data", "tsf-models", "tsf-run", "tsf-check", "tsf-report"]:
        src = Path("agent-skills") / skill / "SKILL.md"
        assert src.read_bytes() == (Path(".codex/skills") / skill / "SKILL.md").read_bytes()
        assert src.read_bytes() == (Path(".claude/skills") / skill / "SKILL.md").read_bytes()


def test_doctor_json_contract():
    out = subprocess.check_output([sys.executable, "-m", "tsf_paperkit.cli", "doctor", "--format", "json"], text=True)
    data = json.loads(out)
    assert {"status", "checks", "environment", "cache"}.issubset(data)
