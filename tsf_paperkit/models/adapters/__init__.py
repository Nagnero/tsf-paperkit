from __future__ import annotations

from tsf_paperkit.models.adapters.amd import ADAPTER_SPEC as AMD_SPEC
from tsf_paperkit.models.adapters.chronos import ADAPTER_SPEC as CHRONOS_SPEC
from tsf_paperkit.models.adapters.moirai import ADAPTER_SPEC as MOIRAI_SPEC
from tsf_paperkit.models.adapters.patchtst import ADAPTER_SPEC as PATCHTST_SPEC
from tsf_paperkit.models.adapters.timemixer import ADAPTER_SPEC as TIMEMIXER_SPEC
from tsf_paperkit.models.adapters.timesfm import ADAPTER_SPEC as TIMESFM_SPEC
from tsf_paperkit.models.adapters.tirex import ADAPTER_SPEC as TIREX_SPEC

ADAPTER_SPECS = {
    spec.key: spec
    for spec in [
        PATCHTST_SPEC,
        TIMEMIXER_SPEC,
        AMD_SPEC,
        TIMESFM_SPEC,
        CHRONOS_SPEC,
        MOIRAI_SPEC,
        TIREX_SPEC,
    ]
}

__all__ = ["ADAPTER_SPECS"]
