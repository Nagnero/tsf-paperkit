from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.moirai",
    display_name="Moirai-style foundation model",
    status="planned-auth-or-license-review",
    optional_extra="foundation-adapters",
    dependency_notes=["Foundation-model runtime dependencies must stay optional."],
    implementation_notes=[
        "Document context length, covariate support, and probabilistic output handling before coding.",
        "Initial integration should emit point forecasts matching BaseForecastModel.predict.",
    ],
    upstream_refs=["Moirai model card/repository to be reviewed before implementation"],
)
