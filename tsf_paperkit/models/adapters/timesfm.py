from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.timesfm",
    display_name="TimesFM-style foundation model",
    status="planned-auth-or-license-review",
    optional_extra="foundation-adapters",
    dependency_notes=["Foundation-model SDKs and weights must remain optional and cache-managed."],
    implementation_notes=[
        "Require explicit user action for any gated weights, license acceptance, or large downloads.",
        "Adapter should support inference-first smoke before any fine-tuning workflow.",
    ],
    upstream_refs=["TimesFM model card/repository to be reviewed before implementation"],
)
