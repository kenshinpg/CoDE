# Supplementary comparison matrices

These aggregate tables accompany the PDF Supplementary Materials. Selection tables use the completed random-control extension. Other tables
retain the corresponding manuscript results.

The four `ddx_audit_*` / `trace_audit_*` files report independent-reviewer targeted
mapping sensitivity. DDXPlus uses 53 explanations / 32 patient-pattern clusters;
TRACE uses 40 explanations / 33 inferred subjects. `encoder` denotes the
Representation probe and `lightgbm` the Tree probe. Tree-derived scores under
the Tree probe are self-evaluation.

R1/R2 denote reviewers. DDXPlus C/I denote conservative/inclusive matching;
TRACE E/C denote explicit/compatible matching. Auto denotes automatic mappings.
Fixed holds automatic recovery outcomes fixed while changing scores. Updated
recomputes outcomes. `failures` counts explanations. Contrast fields give AP
differences followed by unadjusted 95% intervals from 1,000 cluster bootstrap
draws. Positive differences favor the first named score; Association, Local and
Group are coverage scores, and Tree A is direct support.

Sampling was conditioned on original failures. These estimates describe the
targeted audit subset. Main full-test results use automatic citations.

`ddx_descriptive_policy_AP.csv` preserves the descriptive local positive-SHAP
contrasts on the same 53 targeted DDXPlus explanations. Its Association comparator
is direct support A. These entries have no confidence intervals.

`trace_permutation_AP.csv` contains every one of the ten observed-group weight
shuffles for the correct-proposal TRACE analysis (614 explanations / 210
admissions). Failure outcomes and citation fractions were held fixed. The PDF
reports their mean; this is a descriptive control, not a permutation test.

AP differences are in AP units (0 to 1 scale), not accuracy percentage points.
Square brackets inside a CSV cell retain the reported interval endpoints.

## Full selection and controlled-subset matrices

`ddx_selection.csv` and `trace_selection.csv` give gains over each selector's
own fact-count-matched random controls, in accuracy percentage points, at
25/50/75% group budgets. Populations are 1,490 DDXPlus patients and 251 TRACE
admissions. Differences between rows are not paired method contrasts.

`ddx_controlled.csv` and `trace_controlled.csv` give patient/admission-averaged
success-failure concordance percentages and marginal paired 95% intervals.
DDXPlus common sets number 11,641/12,397/10,224 at 25/50/75%; Representation
contributes 835/225/36 patients. TRACE has 2,510 sets from 251 admissions per
budget; Representation contributes 231/182/109 admissions with 4,497/3,185/1,609
discordant pairs, and Tree contributes 178/133/65 with 3,239/2,272/1,006 pairs.
Fixed-count fact coverage ties (50% concordance); R has the same fixed-patient,
fixed-diagnosis ordering as A. DDXPlus controlled Tree self-evaluation was omitted.

R/T denote Representation/Tree probes. Asterisks flag self-evaluation;
Group-enhanced shares the Representation backbone. Cells retain estimate and
interval together. These marginal intervals accompany the main figures; the
simultaneous core comparisons remain in the PDF Supplementary Materials.

## Random-control extension

The primary `ddx_selection.csv` and `trace_selection.csv` matrices use 50
random draws at 25% and ten at 50/75%. Sampling is with replacement; 50 draws
can include repeated sets. Existing self-evaluation markers are retained.

`selection_random_control_stability.csv` reports each dataset, probe, population,
comparison family, selector and patient/admission count. `gain10_pp`/`gain50_pp`
are percentage-point gains over the corresponding random means; `ci50_*` are
marginal 95% patient/subject-bootstrap limits and `change_pp` is gain50 minus
gain10. Blank gain50/CI/change entries with `retained_original_10` identify
TRACE all-test Representation Tree-rule and retrieval families. These whole
families retain ten draws because additional controls encounter unseen states
in two admissions outside the primary population. No patient is removed.

`intervention_control_MC.csv` describes correct-proposal signed local-SHAP
addition/deletion using the ten observed alternatives. Effects, MC standard
errors and limits are in probability percentage points. `conditional_mc_*`
are 2.5/97.5 percentiles from 5,000 resamples within those ten alternatives,
conditional on fixed patients and models. `patient_ci_*` retain the original
patient-bootstrap limits; distinct-control counts describe per-task minimum,
median and maximum. This adds no new interventions.

The TRACE source-fact correction sensitivity retains ten selection controls
in both fact views. Controlled subset sampling, intervention alternatives
and weight permutations retain their original counts.

## Adjudicated citation sensitivity

- `adjudication_mapping_quality.csv`: automatic-versus-reference micro precision, recall and F1, plus extra and missed facts. Metrics use complete adjudicated references; random and failure-targeted strata remain separate. DDXPlus strata overlap by one explanation.
- `adjudication_AP_contrasts.csv`: 72 paired contrasts, in AP units. `automatic` uses original citations/outcomes; `adjudicated_fixed_endpoint` changes citations/scores and retains original outcomes; `adjudicated_updated` recomputes outcomes. `change_*` is the corresponding adjudicated contrast minus the automatic contrast. Simultaneous intervals cover the three contrasts within dataset/audit/probe/proposal strata. Requested, defined and undefined bootstrap counts are explicit; intervals condition on defined replicates. Models and full-input eligibility are fixed.
- `adjudication_endpoint_transitions.csv`: probe outcome/prediction transitions, citation-set changes and disease-level failure counts in the common comparison populations.
- `adjudication_population_flow.csv`: reviewed, reference-complete, eligible and common finite-score counts. Twelve explanations with unresolved reference facts are excluded from both mapping versions.
- `shared_workbook_outcomes.csv` and `shared_workbook_contrasts.csv`: the earlier shared-workbook sensitivity, separate from independent-reviewer agreement and the adjudicated reference. Its intervals are unadjusted 1,000-draw intervals.

`cohd_coverage`, `tree_positive_shap`, `group_coverage` and `lightgbm_absolute` denote Association coverage, local positive-SHAP coverage, uniform group coverage and Tree direct support A. `encoder` is the Representation probe. These tables contain aggregate statistics; protected patient records and review text are excluded. Full-test main results continue to use the automatic citation sets.

## Premise-role sensitivity

`role_sensitivity/` contains the three-set contrasts, endpoint transitions, role frequencies and population flow for the 100-explanation audit. Its README defines outcome versions, paired changes and simultaneous interval families. Full-test tables retain pooled automatic citations.

## Additional TRACE descriptive tables

`trace_exclusion_profile.csv` gives the retained/excluded test-admission profile
before full-input correctness filtering. `trace_independent_review_agreement.csv`
and `trace_independent_review_transitions.csv` retain the original independent
review results (E: explicit; C: compatible). Transition cells report remaining
failures followed by recovered/newly failing counts. The random and targeted
strata use different sampling rules. These tables precede the adjudicated
reference analysis and are reported separately from it.

`trace_natural_paired_contrasts.csv` contains the original correct-proposal paired
AP contrasts under both probes with marginal 95% intervals. The manuscript's
core simultaneous intervals remain in the supplementary PDF. R/T denote
Representation/Tree; Tree-source results evaluated by Tree are self-evaluation.

These four tables reproduce existing v3_3 aggregate results; no new observations
or fitted models were introduced. Current primary selection matrices and figures
use 50 random controls at 25% and ten at 50/75%; the documented all-test exceptions
remain in the stability table.