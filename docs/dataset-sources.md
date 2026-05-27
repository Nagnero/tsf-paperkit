# Dataset Source Ledger

`tsf-paperkit` tracks dataset recipes, not third-party data artifacts. A non-toy dataset can become `status: supported` in `configs/dataset_registry.yaml` only after the machine-readable ledger `configs/dataset_sources.yaml` records an approved `source_approval_id` for that exact recipe/source. This Markdown page is the human-readable summary of that gate.

| Dataset / family | Candidate source | Registry status | Source status | Notes |
| --- | --- | --- | --- | --- |
| `toy_series` | Bundled `examples/toy_series.csv` | supported | approved | Tiny demo fixture for smoke tests only. |
| ETT (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`) | THUML Time-Series-Library `ETT-small` folder | placeholder | pending | Candidate public source; verify license/citation/checksum before automatic download. |
| Electricity / ECL | THUML Time-Series-Library `electricity` folder | placeholder | pending | Candidate public source; larger multivariate CSV. |
| Weather | THUML Time-Series-Library `weather` folder | placeholder | pending | Candidate public source; verify protocol before benchmark use. |
| Traffic | THUML Time-Series-Library `traffic` folder | placeholder | pending | Candidate public source; large, not mandatory for first-pass smoke. |
| Exchange | THUML Time-Series-Library `exchange_rate` folder | placeholder | pending | Candidate for later enablement. |
| ILI / illness | THUML Time-Series-Library `illness` folder | placeholder | pending | Candidate for later enablement. |
| Solar-Energy | TBD | placeholder | pending | Common in TSF papers, but no approved automatic source yet. |
| PeMS03/04/07/08 | TBD | placeholder | pending | Potentially large/license-sensitive; no automatic source yet. |
| Monash tourism monthly | https://forecastingdata.org/ | placeholder | pending | Manual upstream repository; no automatic download in this phase. |

External grounding:
- THUML Time-Series-Library dataset card: https://huggingface.co/datasets/thuml/Time-Series-Library
- THUML Time-Series-Library tree: https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main
- TimeMixer paper, which references common TSF datasets including ETT, Weather, Solar-Energy, Electricity, Traffic, PeMS, and M4: https://arxiv.org/pdf/2405.14616

Policy:
1. Do not commit downloaded datasets, caches, model weights, checkpoints, results, or secrets.
2. Do not mark non-toy recipes `supported` until source/license/citation/size/checksum expectations are reviewed and recorded in `configs/dataset_sources.yaml`.
3. Approved remote recipes must use HTTPS and include a checksum or explicit checksum waiver.
4. Smoke/demo outputs are not benchmark claims.
