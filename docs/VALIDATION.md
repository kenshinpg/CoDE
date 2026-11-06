# Local validation, 6 November 2026

The package was assembled against the accepted manuscript and locally corrected production proof. The manuscript, fitted study models and accepted numerical results were left unchanged.

| Check | Result |
| --- | --- |
| Python source parsing | All included Python files parsed successfully |
| Existing contract tests | 125 run: 122 passed; three skipped |
| Skipped tests | One tiny PyTorch fitting test requires PyTorch, absent from the local QA interpreter; two integration assertions require separately authorized review inputs |
| Public DDXPlus example | Four actual DDXPlus-derived fact sets passed the mapping validator and coverage calculation |
| Patient-free generation/matching contracts | Strict generation, whole-text matching and span-fallback schemas exported successfully |
| Selection estimates/intervals | All 114 exported values agree with full-precision accepted summaries after display rounding |
| Figure 1 | All 34 generated plotting records match the final-proof numerical reference |
| Exploratory DDXPlus distributions | All 60 matching patient-effect distribution means and counts agree with accepted aggregates |
| Figure reconstruction | Three main numeric figures and two Supplementary numeric figures generated as PDF, SVG and PNG |
| Table/artifact coverage | 34 table exports; map covers all 40 main/Supplementary figures and tables |
| Release scans | No detected workstation/storage paths, private service hosts, credential literals or patient-linkage fields in result exports |
| Archive | ZIP integrity and member paths checked; caches, build outputs and private input files excluded |

Tests exercised the supplied scoring/statistical functions on constructed records. Final clinical experiments, licensed source processing and large-model inference were not rerun. Replotting uses accepted aggregate statistics and does not regenerate bootstrap draws from individual clinical records. The small DDXPlus example's optional CSV fitting route requires separately downloaded public files; local validation covered its bundled fact/coverage route.

The clinical construction figure is supplied as an editable SVG and accepted PDF. Font-dependent numeric redraws preserve the source values; their PDF bytes need not match the production figure files. The accepted figures remain available for direct comparison.

`requirements/` describes installation dependencies. Historical model configurations are in the experiment configs and Table S4 export. The package's local QA environment is distinct from the historical GPU serving environment. The original runtime precision/version gaps reported in the paper are retained in the methods specification.

The LANTERN upstream handoff has been accepted and integrated. All 29 manifest-tracked files matched their intended published digests after reversing an archive line-ending conversion; all 29 local historical-source bindings were also checked. Fourteen critical function/class ASTs matched their historical sources. Six model-free boundary checks passed, including the CoDE exporter → original generator input validator → final whole-text matching preparation on 2,400 invented admissions and 9,600 tasks. One deliberately failed synthetic generation remained unavailable. No model, API service or clinical record was used for these checks.

The original ZIP used CRLF shell endings. Included shell scripts now use LF, and `.gitattributes` preserves frozen byte identities during checkout. Both received and integrated source bindings remain documented.

Before public posting, the authors/institution still need to select the license for team-authored/shared code. No additional LANTERN scientific experiment or code handoff is required for the agreed release scope.
