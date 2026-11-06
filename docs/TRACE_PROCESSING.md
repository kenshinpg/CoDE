# TRACE processing specification

## Data lineage

MIMIC-IV → MIMIC-CDM clinical derivative → text/laboratory observations → original concept links and assertion states → frozen visible patient facts F → generated explanations → exact patient-fact citations → CoDE evaluation.

The received clinical derivative covers appendicitis, cholecystitis, diverticulitis and pancreatitis. The source study extracted histories, examinations, first laboratory results and radiology findings. This study's supplied source package was not independently reconciled record-by-record with an official release. The task is retrospective record-based evidence support and uses no common verified pre-diagnosis cutoff.

## Source text and laboratories

EntityCoding released NER/assertion checkpoints identify text observations. Sentence windows contain up to 512 model tokens; assertion context extends up to 100/25 spaCy tokens to the left/right within the sentence. HPO Phenotypic abnormality descendants supply the concept vocabulary. The SapBERT-derived encoder `pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb` uses attention-mask mean pooling and normalized 768-dimensional embeddings, a 100-token limit, and 20 unique candidate concepts per query.

Locally deployed Qwen3.8-27B selects a candidate concept or null for history/examination mentions and permits eligible multiple links for radiology units. Temperature is zero and thinking is disabled. Original links and model-review suggestions remain separate. Labs use source reference bounds to produce LOW/WITHIN_RANGE/HIGH, with no unit conversion or external reference replacement.

The source extraction/linking orchestration is maintained by LANTERN and is included under `upstream/TRACE_processing`. Its source pipeline covers history/examination NER and assertion, native-tokenizer HPO retrieval/linking, model review, radiology units/linking, laboratory states and the four-source export. The original 4B full-generation and final whole-text matching drivers are included alongside their complete prompts/schemas and historical bindings. Read that supplement's README, `docs/CONTRACTS.md` and `docs/ORIGINAL_BINDING.md` for exact commands and resource requirements. Clinical inputs, weights and external ontology resources are obtained separately.

## Frozen CoDE input

`prepare_trace2400_view.py` and `src/code_eval/trace2400/view.py` implement original-link selection, present/absent assertion eligibility, radiology parent units, conflict exclusion, duplicate concept collapse and diagnosis-name screening. Train-only feature eligibility requires two subjects. The frozen dictionary has 1,595 features; explicit absence is an observed value. Split seed 261013 gives 1,680/361/359 train/development/test admissions, grouped by the available inferred-subject identifier. There are 2,383 inferred subjects. Subject inference was not independently validated against an official crosswalk.

`configs/trace2400_view_v2_names.json` supplies the exact policy. `scripts/export_trace_lantern_handoff.py` and `configs/trace_lantern_handoff/` export the accepted view into the frozen production input: outcome-free split membership, patient display facts, vocabulary, 9,600 generation tasks and 80 development pilot tasks. Reference outcomes remain in a separate sidecar. The original generator validates these files before use. Later human source-fact corrections form a separate sensitivity view and leave the original main-analysis view intact.

## Explanation generation and citation matching

Qwen3-4B generated four explanations per admission from displayed F only. The request returns an English explanation and three distinct ranked diagnoses, treats unlisted findings as unknown, and omits reference labels, raw source text and provenance. Generation uses temperature 0.7, top-p 0.8, up to 2,048 output tokens and a service context of 8,192 tokens.

The second-generator experiment uses Qwen3.8-27B on the full fixed 359-admission test partition, four explanations each. `scripts/trace_generator_rq2/original_generation.py` preserves the generation prompt/schema; `base.py` and `segment.py` preserve patient-fact matching and span fallback. `service.py` executes those contracts. Exact-match handling, fallback and retry policy are visible in code. The historical full-production retry policy and the second-generator infrastructure-recovery policy are distinct; no transport-level equivalence is assumed.

Matching uses up to eight facts per request, temperature zero, disabled thinking, a 65,536-token service context and up to 3,072 output tokens; span fallback permits 2,048 tokens. Format retries use the configured temperature 0.2. Matching requires verbatim spans and compatible feature/value/assertion decisions. Clinical examples, cached outputs and local service addresses are excluded from the public package.
