# Future model adapters

These files are **metadata placeholders**, not runnable implementations. They exist so future contributors and agents can add advanced TSF models without changing the MVP harness contract.

Adapter promotion checklist:

1. Keep `BaseForecastModel` shape contract: input `batch, seq_len, channels`; output `batch, pred_len, channels`.
2. Keep dependencies optional via `pyproject.toml` extras.
3. Keep weights/checkpoints out of git; use `.cache/tsf-paperkit/models` or `TSF_PAPERKIT_MODEL_CACHE`.
4. Record provider, revision, license/auth note, expected files, and cache key in `configs/model_registry.yaml`.
5. Add a tiny CPU smoke test before changing a registry entry from `future-adapter` to runnable.
6. Do not claim benchmark/SOTA from smoke configs.

Current placeholders: PatchTST, TimeMixer, AMD-style decomposition, TimesFM, Chronos, Moirai, TiRex.
