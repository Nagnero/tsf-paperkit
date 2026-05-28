from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForecastAdapterSpec:
    """Metadata-only contract for future forecasting adapters.

    The MVP does not implement advanced architectures here. The spec records the
    shape, dependency, asset, and policy expectations that a future adapter must
    satisfy before it can be promoted from `future-adapter` to runnable code.
    """

    key: str
    display_name: str
    status: str
    optional_extra: str | None
    input_shape: str = "batch, seq_len, channels"
    output_shape: str = "batch, pred_len, channels"
    fit_contract: str = "fit(train_loader, val_loader, config) must train or adapt without mutating data splits"
    predict_contract: str = "predict(test_loader) must return (y_pred, y_true) numpy arrays with output_shape"
    asset_policy: str = "do not commit weights/checkpoints; use model cache and upstream licenses"
    dependency_notes: list[str] = field(default_factory=list)
    implementation_notes: list[str] = field(default_factory=list)
    upstream_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "status": self.status,
            "optional_extra": self.optional_extra,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "fit_contract": self.fit_contract,
            "predict_contract": self.predict_contract,
            "asset_policy": self.asset_policy,
            "dependency_notes": list(self.dependency_notes),
            "implementation_notes": list(self.implementation_notes),
            "upstream_refs": list(self.upstream_refs),
        }


def require_optional_dependency(package: str, extra: str, adapter_name: str) -> None:
    """Raise an actionable error for adapter-only dependencies.

    Placeholder adapters call this once they become real implementations instead
    of making core `pip install -e .` depend on every research stack.
    """

    try:
        __import__(package)
    except ImportError as exc:  # pragma: no cover - exercised through message contract tests later
        raise ImportError(
            f"{adapter_name} requires optional dependency {package!r}. "
            f"Install it with `pip install -e '.[{extra}]'` or choose a built-in MVP model."
        ) from exc
