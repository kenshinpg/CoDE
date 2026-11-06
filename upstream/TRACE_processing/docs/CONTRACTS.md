# Data contracts and complete prompt/schema locations

All identifiers below are field names or construction rules, not real patient examples.

| Layer | Main keys | Meaning |
|---|---|---|
| Received source | disease-named PKL -> admission key -> Patient History / Physical Examination / Laboratory Tests / Reference Range Lower / Reference Range Upper / Radiology | Received dictionaries; filename disease labels stay outside clinical model inputs |
| EntityCoding | document_id, patient_id, source_field, text, entities[] | patient_id is a legacy admission key; entity includes mention_id, raw_start/end/quote, entity_type, assertion, status |
| History/exam HPO link | mention_id, entity, mapping.hpo_id, candidates, context | One candidate concept or null; assertion is independent of concept linking |
| Review export | mention_id, original mapping, audit_state/flags, clinical_correction_applied=false | Suggestions and incomplete tasks remain explicit, no automatic rewrite |
| Source document | document_id -> admission_id; source_type, raw_text, source_locator, native_metadata, time | Original fields/list positions preserved; list order is not clinical chronology |
| Observation | observation_id -> admission_id/document_id; raw, concept/concept_links, value, attributes, time, representation_group_id, derived_from_observation_ids | One source mention or native lab snapshot; null document_id is allowed for labs |
| Lab fact view | admission_id, view_id, facts[] | fact_id, LAB feature_id, LOW/WITHIN_RANGE/HIGH, display_text, source_observation_ids and shared representation_group_id |
| Final matching | task_id, patient_id, decisions[], operational_exact_fact_ids, visible_fact_scan_status | All visible facts assessed; E(r) is deduplicated exact fact IDs subject to analysis availability |

Raw spans are zero-based Unicode character offsets, end exclusive. The original
assertions PRESENT/ABSENT/FAMILY/HYPOTHETICAL/POSSIBLE are preserved and normalized
to lower-case values in attributes. Unknown attributes remain null with status/basis.
A non-null HPO link does not imply an affirmative current patient finding. Null link
means unlinked, not absent. within-range only describes the supplied reference bounds.
Labs use finite exact x,L,U, L<=U; x<L LOW, x>U HIGH, otherwise WITHIN_RANGE.
The main lab policy assumes paired source values and bounds share a scale when units
are not independently supplied; it does not claim validated dimensional equivalence.
No item-specific severity bins or implicit unit conversion are introduced.

Complete SYSTEM text, validators and Pydantic schemas are in the source files below.
They are not truncated excerpts or real populated clinical requests.

| Stage | Prompt / schema definitions |
|---|---|
| NER and assertion | scripts/entitycoding_v1/run.py adapter; vendor/entity-coding/ner executable and rule files; published NER/AC checkpoints obtained separately |
| History/exam HPO link | scripts/rag_hpo_link_v2/link.py: SYSTEM, Mapping, candidate_only |
| History/exam review | scripts/hpo_review_full_v1/review.py: COMMON, EXTRACTION, MAPPING and its Pydantic classes |
| Radiology decomposition | scripts/radiology_atoms_v1/atoms.py: SYSTEM, Atom, Response, validate |
| Radiology map/review | scripts/radiology_mapping_v1/pipeline.py: MAP_SYSTEM, REVIEW_SYSTEM, MapResponse, ReviewResponse, validate |
| Qwen3-4B generation | scripts/trace_rollouts_full_v1/pipeline.py: SYSTEM, Answer, messages, parse_response |
| Final free matching | scripts/trace_match_full_v1/base.py: PROMPT, Quote, Decision, Answer, validate |
| Failed-group fallback | scripts/trace_match_full_v1/segment.py: PROMPT, Decision, Answer, segments, validate |

Reference source schema for lab metadata: an array of objects containing string
itemid and label, optional fluid/category/corresponding_ids. tools/prepare_local_resources.py
produces it from the authorized source CSV. Clinical observation schema is implemented
by trace2400_unified_v1/code_contracts.py; it verifies associations, spans and dependency
integrity, not clinical correctness. Exact output field definitions are in the retained
builders; public runtime IDs and code hashes will differ after portability edits.
