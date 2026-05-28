from __future__ import annotations

from tsf_paperkit.models.adapter_contract import ForecastAdapterSpec

ADAPTER_SPEC = ForecastAdapterSpec(
    key="future.patchtst",
    display_name="PatchTST",
    status="planned",
    optional_extra="patchtst",
    dependency_notes=["Likely needs PyTorch plus tensor rearrangement utilities such as einops."],
    implementation_notes=[
        "Wrap a local implementation or a clearly licensed upstream implementation behind BaseForecastModel.",
        "Keep patch length, stride, channel-independence, and head settings in model params.",
        "Add a tiny CPU smoke test before promoting the registry entry to runnable.",
    ],
    upstream_refs=["PatchTST paper/repository to be reviewed before implementation"],
)
