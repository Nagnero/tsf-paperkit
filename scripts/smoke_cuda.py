#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys

import torch

if not torch.cuda.is_available():
    print(json.dumps({"status": "skipped", "reason": "torch.cuda.is_available() is false"}))
    raise SystemExit(0)
cmd = [sys.executable, "-m", "tsf_paperkit.cli", "run", "--config", "configs/example.yaml"]
# The example uses device:auto, which resolves to CUDA when available.
res = subprocess.run(cmd, text=True, capture_output=True)
print(json.dumps({"status": "pass" if res.returncode == 0 else "fail", "command": cmd, "stdout": res.stdout, "stderr": res.stderr}))
raise SystemExit(res.returncode)
