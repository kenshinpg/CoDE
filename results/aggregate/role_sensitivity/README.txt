Premise-role sensitivity: aggregate data

The audit contains 50 explanations per cohort: 30 random and 20 failure-targeted.
All refers to all adjudicated record-compatible citations; first is the union of
support and opposition for the first-ranked proposed diagnosis; support contains
supporting citations alone. The diagnosis target and full-record weights are fixed.

three_comparisons_PRIMARY.csv: correct proposals with Representation probe.
three_comparisons_all_strata.csv: both probes and correct/incorrect/all proposals.
Differences and intervals are in AP units. restricted_updated recomputes outcomes;
restricted_fixed_all_endpoint retains All outcomes. change is restricted minus All
for the same comparison and common records across all three sets. Bootstrap
counts distinguish requested, defined and undefined draws. Simultaneous families
contain three contrasts within each panel and outcome version.

endpoint_transitions.csv: common-population outcome and prediction transitions.
role_census.csv: reviewed population, citation-set sizes and textual role counts.
population_flow.csv: eligibility and common score availability.
set_endpoint_descriptives.csv: subset sizes, scores and endpoint frequencies.

cohd_coverage = Association coverage; tree_positive_shap = local positive-SHAP
coverage; group_coverage = uniform group coverage; lightgbm_absolute = Tree A;
encoder = Representation probe. patients denotes patients/admissions, while
clusters uses DDXPlus complete-pattern groups or TRACE subjects. Empty sets retain
demographics and remain evaluable. Random and targeted samples are separate.
Counts of opposing and supporting citations can overlap. Argumentative use of a
negative finding describes textual use, without judging clinical correctness.
All files contain aggregate results; protected records and review text are omitted.