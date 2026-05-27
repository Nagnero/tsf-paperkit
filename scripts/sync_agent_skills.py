#!/usr/bin/env python
from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "agent-skills"
TARGETS = [ROOT / ".codex" / "skills", ROOT / ".claude" / "skills"]
SKILLS = ["tsf-doctor", "tsf-data", "tsf-models", "tsf-run", "tsf-check", "tsf-report"]
APPROVED = {
    "tsf-doctor": ["doctor", "smoke_cuda.py", "sync_agent_skills.py", "run --config"],
    "tsf-data": ["data list", "data prepare", "dataset_registry.yaml"],
    "tsf-models": ["model list", "model prepare", "model_registry.yaml"],
    "tsf-run": ["run --config", "ablate --config"],
    "tsf-check": ["check --config"],
    "tsf-report": ["table --input", "check --config"],
}


def skill_dirs(base: Path):
    return [base / s for s in SKILLS]


def validate_skill(path: Path) -> list[str]:
    errors = []
    text = (path / "SKILL.md").read_text()
    if not text.startswith("---\n") or "name:" not in text.split("---", 2)[1] or "description:" not in text.split("---", 2)[1]:
        errors.append(f"{path}: missing portable frontmatter")
    front = text.split("---", 2)[1]
    keys = [line.split(":", 1)[0].strip() for line in front.splitlines() if ":" in line]
    if any(k not in {"name", "description"} for k in keys):
        errors.append(f"{path}: frontmatter keys must be name/description only")
    if "/root/workspace" in text:
        errors.append(f"{path}: contains absolute local path")
    allowed = APPROVED[path.name]
    for cmd in re.findall(r"python -m tsf_paperkit\.cli ([^`\n]+)", text):
        if not any(a in cmd for a in allowed):
            errors.append(f"{path}: unapproved CLI entrypoint {cmd}")
    if "```python" in text and len(text) > 3500:
        errors.append(f"{path}: likely duplicates deterministic package logic")
    return errors


def sync(check: bool = False) -> int:
    errors = []
    for sdir in skill_dirs(SRC):
        errors.extend(validate_skill(sdir))
    if errors:
        print("\n".join(errors))
        return 1
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for skill in SKILLS:
            src = SRC / skill
            dst = target / skill
            if check:
                if not (dst / "SKILL.md").exists() or not filecmp.cmp(src / "SKILL.md", dst / "SKILL.md", shallow=False):
                    print(f"drift: {dst}")
                    return 1
            else:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        manifest = {"source": "agent-skills", "skills": SKILLS, "note": "SKILL.md copies are byte-for-byte; this sidecar is provenance only."}
        if not check:
            (target / ".sync-manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    raise SystemExit(sync(parser.parse_args().check))
