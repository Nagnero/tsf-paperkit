from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.chronos",
    display_name="Chronos-style foundation model",
    status="planned-auth-or-license-review",
    optional_extra="foundation-adapters",
    dependency_notes=["Transformer/foundation-model dependencies must remain optional."],
    implementation_notes=[
        "Keep tokenization/preprocessing inside the adapter without changing dataset windowing.",
        "Do not auto-download weights during import or model listing.",
    ],
    upstream_refs=["Chronos model card/repository to be reviewed before implementation"],
)
