# Validation

Local checks used Python 3.12 on Windows. GPU inference and clinical experiments were not rerun.

| Check | Result |
| --- | --- |
| Existing contract tests | 125 run: 122 passed; three skipped |
| Skipped tests | One requires PyTorch; two require separately authorized audit inputs |
| DDXPlus example | Bundled coverage calculation and CSV fitting route checked; empty and unsupported test inputs produce diagnostic counts and a nonzero exit status |
| Selection estimates and intervals | All 114 exported values match the accepted summaries after display rounding |
| Main overview figure | All 34 plotting records match the final-proof numerical reference |
| DDXPlus edit distributions | Counts and means match the accepted aggregates for all 60 distributions |
| Figure generation | Three main numeric figures and two supplementary numeric figures rendered as PDF, SVG and PNG |
| Artifact map | Covers all 40 main and supplementary figures and tables, including 34 table exports |
| TRACE input boundary | Six model-free checks passed, including export, generation-input validation and final-matching preparation on synthetic admissions |

Run the public checks with:

```bash
python scripts/verify_aggregates.py
python scripts/plot_paper.py
python examples/run_ddxplus.py
# Requires the experiment dependencies:
python -m unittest discover -s tools/CoDE/tests -v
```

The tests use constructed fixtures. Aggregate checks and figure generation verify the supplied estimates; reproducing patient-cluster bootstrap inference requires the original evaluation inputs. CSV fitting requires official DDXPlus files obtained separately. Fresh environment installation and Linux/GPU execution were not retested locally. See [EXPERIMENTS.md](EXPERIMENTS.md) for experiment inputs and dependencies.
