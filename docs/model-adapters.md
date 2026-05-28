# Model Adapter Contract

`tsf-paperkit` keeps the MVP small: runnable models are `naive`, `linear`, and `dlinear`. Advanced TSF models are tracked as **future adapters** until their dependencies, source, license, weight policy, and smoke tests are explicit.

## Runtime contract

Every runnable adapter must implement `BaseForecastModel`:

```python
fit(train_loader, val_loader, config)
predict(test_loader) -> tuple[y_pred, y_true]
save(path)
load(path)
num_params() -> int
```

Shape contract:

```text
input x:  batch, seq_len, channels
output y: batch, pred_len, channels
```

Adapter rules:

1. Do not change shared dataset splitting, train-only scaling, or windowing semantics.
2. Do not download weights or datasets during import, `model list`, or test collection.
3. Keep non-core dependencies in optional extras from `pyproject.toml`.
4. Keep weights/checkpoints out of git; use `.cache/tsf-paperkit/models` or `TSF_PAPERKIT_MODEL_CACHE`.
5. Record provider, revision, license/auth, expected files, cache key, and adapter key in `configs/model_registry.yaml`.
6. Add a tiny CPU smoke test before promoting a future adapter to runnable.
7. Smoke outputs are readiness checks only, not benchmark/SOTA claims.

## Optional extras

```bash
pip install -e '.[patchtst]'
pip install -e '.[research-adapters]'
pip install -e '.[foundation-adapters]'
pip install -e '.[adapters]'
```

These extras only prepare dependency lanes. They do not make future adapters runnable by themselves.

## Future adapter inventory

| Registry name | Adapter key | Optional extra | Current state |
|---|---|---|---|
| `patchtst_future` | `future.patchtst` | `patchtst` | planned; dependency lane and placeholder spec only |
| `timemixer_future` | `future.timemixer` | `research-adapters` | planned; source/license review required |
| `amd_future` | `future.amd` | `research-adapters` | planned; exact AMD-style variant must be selected |
| `timesfm_future` | `future.timesfm` | `foundation-adapters` | planned; auth/license/weight review required |
| `chronos_future` | `future.chronos` | `foundation-adapters` | planned; auth/license/weight review required |
| `moirai_future` | `future.moirai` | `foundation-adapters` | planned; auth/license/weight review required |
| `tirex_future` | `future.tirex` | `foundation-adapters` | planned; auth/license/weight review required |

## Promotion checklist

To promote `patchtst_future` or another future adapter:

1. Select an upstream source and record the exact revision.
2. Verify license and citation notes.
3. Add optional dependencies only to the relevant extra.
4. Implement a class under `tsf_paperkit/models/` or `tsf_paperkit/models/adapters/` that implements `BaseForecastModel`.
5. Add the adapter key to `MODEL_CLASSES` only after the implementation works.
6. Add a fixture/smoke test with a tiny run on `toy_series` or a prepared public dataset.
7. Update `configs/model_registry.yaml` from `kind: future-adapter` to a runnable kind/status.
8. Update README and agent skills with the exact user-facing command.

## Placeholder files

The files in `tsf_paperkit/models/adapters/` intentionally contain metadata, not model code. They are guardrails for future agents and contributors.
