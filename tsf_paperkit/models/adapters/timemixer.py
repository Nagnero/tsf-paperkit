from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.timemixer",
    display_name="TimeMixer",
    status="planned",
    optional_extra="research-adapters",
    dependency_notes=["Keep research-code dependencies optional and isolated from core install."],
    implementation_notes=[
        "Define multi-scale input handling without changing the shared dataloader contract.",
        "Record any upstream code/license constraints in configs/model_registry.yaml before enablement.",
    ],
    upstream_refs=["TimeMixer paper/repository to be reviewed before implementation"],
)
