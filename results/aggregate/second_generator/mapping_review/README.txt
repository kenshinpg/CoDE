Second-generator mapping review and paired replay

R02 personally checked 32 fixed, disease-balanced explanations against patient facts and visible automatic mappings, with scores hidden. Twenty-seven explanations were resolved; five retained uncertain decisions and were excluded symmetrically. This is a single-reviewer sensitivity reference.

The primary panel comprises ten correct-proposal explanations from ten subject clusters, with two Representation-probe failures under either mapping. Automatic, reviewed_fixed_endpoint and reviewed_updated_endpoint refer to the original mapping, reviewed scores with original outcomes, and reviewed scores with recomputed outcomes. Comparisons use identical records and all four required scores within each panel. The full-test results remain unchanged.

AP differences are in raw AP units. Intervals use 5,000 subject-cluster draws shared across versions. For the primary audit panel, 539 no-failure draws have undefined AP; intervals use the remaining 4,461. The maximum-centered-deviation simultaneous intervals cover each family of three contrasts and are reported without clipping to the parameter range. Mapping-change intervals are descriptive marginal paired intervals. Empty fields indicate undefined statistics. The correct-proposal Tree panel has no failures, so its AP is undefined.

three_comparisons_PRIMARY.csv: three contrasts in each of the three mapping/outcome versions.
three_comparisons_ALL.csv: additional incorrect/all-proposal and Tree-probe panels.
paired_contrast_changes.csv: reviewed-minus-automatic changes in each contrast.
population_summary.csv: eligibility, shared denominator, clustering and outcome-transition counts.
score_change_summary.csv: available explanations, changed scores and magnitudes, counted once per explanation.

Across all resolved records, Representation was scorable in 23 and Tree in 27. One Representation success became a failure outside the full-input-eligible population; Tree outcomes did not change. No patient identifiers or clinical text are included in these aggregate files.