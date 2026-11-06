# Release boundary

This local candidate contains the actual CoDE implementation, the accepted LANTERN upstream processing code and final accepted aggregate results. Its runnable public path covers fact validation, coverage scoring, contract tests and figure reconstruction. Full study execution starts from frozen producer exports, trained artifacts and the original split/input manifests described in INPUT_CONTRACTS.

The following are obtained separately:

- Complete DDXPlus study selection/generation/mapping inputs; original public DDXPlus files alone do not recreate a fixed LLM-generated study corpus.
- Credentialed MIMIC-CDM/TRACE records, admission/subject associations, generated explanations, exact mappings, review workbooks and model checkpoints.
- Author/institution approval of the license for newly authored and team-shared code.

`docs/paper_artifact_map.csv` distinguishes figure reconstruction from recomputation on original records. No missing private file is silently replaced with simulated study output. Public fixture examples and constructed test inputs remain separate from reported experiments.
