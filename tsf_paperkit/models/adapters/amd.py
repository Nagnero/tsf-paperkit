from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.amd",
    display_name="AMD-style adaptive multi-scale decomposition",
    status="planned",
    optional_extra="research-adapters",
    dependency_notes=["Keep any decomposition-specific dependencies optional."],
    implementation_notes=[
        "Do not conflate this with MVP DLinear; document the exact adaptive decomposition variant before coding.",
        "Expose decomposition settings through model params and resolved configs.",
    ],
    upstream_refs=["AMD-style model source to be selected and reviewed before implementation"],
)
