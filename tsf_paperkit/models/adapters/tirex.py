from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.tirex",
    display_name="TiRex-style foundation model",
    status="planned-auth-or-license-review",
    optional_extra="foundation-adapters",
    dependency_notes=["Keep TiRex-specific runtime and weights outside core install."],
    implementation_notes=[
        "Confirm upstream availability, license, expected input frequency handling, and weight size first.",
        "Promote only after a selected model prepare path and smoke test exist.",
    ],
    upstream_refs=["TiRex model card/repository to be reviewed before implementation"],
)
