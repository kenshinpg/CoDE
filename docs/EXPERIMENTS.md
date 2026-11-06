# Experiment entrypoints

The implementations below are the actual study runners. Run them from `tools/CoDE` after restoring the corresponding authorized frozen inputs. The local release checks cover constructed fixtures, a public DDXPlus example and aggregate figure reconstruction; they do not rerun the full clinical experiment.

| Purpose | Entry point | Inputs / outputs |
| --- | --- | --- |
| DDXPlus frozen split | `prepare_code10k.py` | Prepared 10k/40k producer export → grouped partitions and citation ledger |
| DDXPlus fitted sources | `run_code10k_sources.py` | Train/development → association, Tree and neural sources |
| DDXPlus RQ1 | `run_code10k_rq1.py` | Frozen models → selections, matched controls, paired accuracy |
| DDXPlus RQ2 | `run_code10k_rq2.py` | Citations + frozen models → failure-AP summaries |
| DDXPlus RQ3 | `run_code10k_rq3.py` | Controlled intact-group subsets → concordance |
| DDXPlus edits/versions | `run_code10k_interventions.py`, `analyze_code10k_intervention_effects.py`, `analyze_code10k_versions.py`, `analyze_code10k_versions_bothcorrect.py`, `analyze_continuous_support.py` | Evidence additions/deletions and fixed-diagnosis support comparisons |
| Random-control extension | `extend_rq1_random_controls.py`, `summarize_random50_parallel.py`, `intervention_control_mc.py` | Reuse fitted models; 50 draws at 25%, ten at higher selection budgets; fixed-family availability rules retained |
| TRACE input/fitting | `prepare_trace2400_view.py`, `run_trace2400_sources.py`, `run_trace2400_extended.py` | Accepted producer export → frozen F and sources |
| TRACE RQ1–RQ3 | `run_trace2400_evaluation.py`, `run_trace2400_rq2.py`, `run_trace2400_tabr_repair.py` | Frozen sources, mappings and controlled subsets → accepted panels |
| Adjudication | `prepare_adjudication_reference.py`, `run_adjudication_sensitivity.py` | Fixed policy/decisions; automatic and adjudicated scores/endpoints on common records |
| Roles | `run_round1_role_sensitivity.py` | All citations, first-diagnosis-related and supporting subsets |
| Core intervals/exclusions | `run_round1_followup.py`, `run_round1_cached_statistics.py` | Fixed scores → bootstrap families, disease deletion, eligibility profiles |
| Budget/calibration | `run_round1_budget_calibration.py`, `run_round1_temperature.py` | Fixed cache + development temperatures → calibration/ranking sensitivity |
| Extra baselines | `run_round1_extra_baselines.py` | Frequency/IDF, Greedy, SIS, Anchors using fitted sources |
| Source correction | `run_trace_F_repair.py`, `summarize_trace_F_repair.py` | Approved limited actions; unchanged original main view |
| Second generator | `run_trace_generator_rq2.py`, `trace_generator_rq2/sharded.py` | Local generation/matching; score/statistics from the original fitted models |
| 32-case matching review | `run_trace27b_review32.py` | 27 determined cases replayed; five uncertain cases separate |

Every runner exposes its exact CLI via `python scripts/<entrypoint>.py --help`. Restore its upstream artifacts before execution. Some server orchestrators require Linux; the pure functions and aggregate plotting run on Windows or Linux. CPU process counts and progress reporting are CLI-controlled where supplied. Set BLAS threads per worker to avoid oversubscription.

The core comparison families were specified after review. Reported intervals retain their actual multiplicity handling, cluster units and conditional-on-fitted-model interpretation. The release does not recast post-review comparisons as prospective hypotheses or combine random and targeted audit samples.

## Running contract tests

```bash
python -m pip install -r requirements/experiments.txt
python -m unittest discover -s tools/CoDE/tests -v
```

Tests use constructed fixtures and local temporary directories; they make no clinical-data request and need no API key. Tiny model-fit tests check implementation contracts and do not reproduce final study effect sizes.

## TRACE source production and original rollout boundary

Use a separate accelerator-compatible environment for `upstream/TRACE_processing`; its requirements preserve the recorded text-processing environment and should not be blindly merged into an existing vendor Torch installation. Run `run_source_pipeline.sh` by named stage as documented there. Its final `outputs/unified` is consumed by the CoDE view constructor.

For an accepted frozen study view, from `tools/CoDE`:

```bash
python scripts/export_trace_lantern_handoff.py --view artifacts/trace2400_view_v2_names --out artifacts/trace_rollout_export --rules configs/trace_lantern_handoff
```

Then set `FROZEN_INPUT` to the export's `production_PRIVATE` subdirectory and use the upstream `run_original_rollouts.sh` commands. The exporter excludes reference outcomes from production messages and retains them separately for evaluation. Final matching can use either the authorized historical source release or the included `prepare_match_source.py` adapter.

The supplied `accepted_view.json` binds the original accepted view. A freshly reconstructed view with new runtime metadata is a new artifact: verify and freeze its own input identities in a separate rules directory before exporting. Historical hashes are not an instruction to silently relabel a fresh rebuild as the original study output. The missing official-source equivalence and runtime provenance described in DATA_ACCESS still apply.
