---
name: tsf-doctor
description: Use when the user wants to verify Linux, Python, PyTorch, CUDA, cache, and Codex/Claude skill readiness for tsf-paperkit.
---
# tsf-doctor

Goal: verify a cloned Linux repo is ready to run tsf-paperkit.

Workflow:
1. Run `python -m tsf_paperkit.cli doctor --format json`.
2. Run `python scripts/sync_agent_skills.py --check`; if it fails, run `python scripts/sync_agent_skills.py` then check again.
3. Run `python -m tsf_paperkit.cli run --config configs/example.yaml` for CPU/auto smoke.
4. Run `python scripts/smoke_cuda.py`; CUDA absence is a structured skip, not failure.
5. Summarize pass/warn/fail entries and repair guidance.

Rules:
- Linux only in v1.
- Do not install NVIDIA drivers or configure secrets.
- Do not claim GPU performance; CUDA smoke only proves one toy run.
