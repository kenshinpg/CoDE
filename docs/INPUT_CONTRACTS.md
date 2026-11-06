# Required input contracts

Run original entrypoints from `tools/CoDE`. Relative paths resolve there. Parameter defaults identify historical artifact basenames; explicit CLI paths may be supplied where supported.

| Stage | Required structure | Produced structure |
| --- | --- | --- |
| DDXPlus cohort | `inputs/ddxplus_sample/data/patients_public.jsonl`, evidence dictionary, fixed selected-case/response records; `inputs/ddxplus_mapping/cosi_sets_40000.jsonl`, `citation_repairs.jsonl` and acceptance manifest | `artifacts/code10k_protocol_v1/patient_split_PRIVATE.jsonl`, `explanation_ledger_PRIVATE.jsonl`, cohort report |
| DDXPlus fitting | Above sample and frozen protocol | Train/development structures, fitted sources, encoding and fitted-source manifest |
| TRACE accepted producer export | Labels, admission-to-subject grouping, source documents, observations, laboratory states/dictionary, acceptance report and manifest | Frozen `F`, train/development/test structures, display facts, feature dictionary and source bindings |
| Natural explanation scoring | Frozen patient facts, generator responses, exact-citation ledger and source/probe artifacts | Explanation-level scores, availability, proposed/reference labels and probe outcomes |
| Adjudication / role sensitivity | Fixed audit membership, confirmed decisions/roles, original scores and models | Same-record paired contrasts and endpoint transitions |
| Plotting | Accepted aggregate panels and comparison matrices | Figures; no individual records required |

A structural model record has `patient_id`, `partition`, `label`, `tokens`, `context` and grouping identifiers. `label` is used for training/reference evaluation and stays outside generated patient messages. DDXPlus context fixes age and sex outside the evidence budget; TRACE context is empty. Observed state tokens encode feature and value. Missing observations contribute no token.

Citation sets contain unique fact IDs present in the same frozen visible record. Unknown, unresolved and technically failed mappings retain their statuses. An unresolved set is distinct from a confirmed empty citation set. Source support and endpoint availability are tracked separately.

For groups B, a group citation fraction is |E intersect b|/|b|. Uniform group coverage averages those fractions; importance-weighted coverage weights them by nonnegative group importance. Direct support A evaluates the cited set jointly; referenced support R subtracts full-record support. An unobserved or unavailable score remains unavailable.

The original runners validate input/model hashes. Frozen study inputs satisfy fixed census and identity checks. New cohorts need a separately frozen protocol and matching manifests; bypassing checks is not a reproduction of the accepted study. Public miniature examples exercise the underlying functions rather than impersonating those frozen inputs.
