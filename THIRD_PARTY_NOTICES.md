# License scope and third-party components

The root MIT License covers original project code and its software documentation. It does not replace third-party licenses. In particular, `tools/CoDE/src/code_eval/tabr_official_core.py` includes the TabR Model adaptation and retains the TabR attribution and MIT terms recorded under `tools/CoDE/vendor/tabr`. Dataset and model access terms are separate.

| Component | Source | Terms / changes |
| --- | --- | --- |
| Sufficient Input Subsets | https://github.com/google-research/google-research/tree/master/sufficient_input_subsets | Apache-2.0; original source and license retained under `vendor/sis` |
| Anchors | https://github.com/marcotcr/anchor | BSD-2-Clause; original license retained under `tools/CoDE/vendor/anchor`. The local adapter removes an unused scikit-learn import. |
| TabR Model class | https://github.com/yandex-research/tabular-dl-tabr | MIT; license retained under `vendor/tabr`. `tabr_official_core.py` supplies exact PyTorch nearest-neighbor lookup and compatibility utilities for TabR-S. |
| Shared producer contracts and TRACE processing | `tools/CoDE/vendor/lantern`; `upstream/TRACE_processing` | Team-authored code is covered by the root MIT License. Third-party source files and data retain the exceptions listed here. |
| EntityCoding | https://github.com/jd4501/entity-coding | Upstream NER/cleaning and generic rules are retained with their MIT notice under `upstream/TRACE_processing/vendor/entity-coding`. Weights are obtained separately. |
| DDXPlus example data | https://github.com/mila-iqia/ddxplus | CC BY 4.0; sample and preprocessing attribution in `examples/DATA_SOURCE.md` |

Model weights and the HPO vocabulary are distributed by their respective providers and are absent from this package. Their names in code identify the actual implementation and do not transfer a model/data license.
