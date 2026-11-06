# DDXPlus example attribution

The four fact sets in `ddxplus_facts.json` are a compact DDXPlus-derived development fixture already used in this project. They contain synthetic patient findings and group/feature definitions. Original record identifiers, natural-language rollouts, human annotations and internal provenance fields have been removed. These four examples are distinct from the frozen 10,000-patient formal evaluation and carry no final model-performance claims. The original downloaded DDXPlus version binding for this old fixture was not retained.

Original dataset: Fansi Tchango et al., **DDXPlus: A New Dataset for Automatic Medical Diagnosis** (NeurIPS 2022). Official repository: https://github.com/mila-iqia/ddxplus. English dataset: https://doi.org/10.6084/m9.figshare.22687585. Dataset terms: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/. Changes: conversion to fact/group JSON, removal of identifiers and selection of four records for interface demonstration.

Default command: `python examples/run_ddxplus.py`. It computes uniform group coverage for a fixed illustrative citation rule and uses the actual CoDE mapping validator.

To fit a small actual association model, obtain the official English training/test CSVs, then run:

```bash
python examples/run_ddxplus.py --train release_train_patients.csv --test release_test_patients.csv --limit 2048
```

This optional CSV route groups observed tokens by their feature code for a compact illustration. Formal study groups come from the frozen producer dictionary and include the full dependency constraints. Paper-scale fitting and evaluation use the original runners and input contracts.
