# TRACE source processing code supplement

This package supplies the missing source-processing segment of the CoDE release:
authorized collaborator-provided MIMIC-CDM derivative -> history/examination
entities and assertions -> HPO links and model-review records -> radiology units
and links -> reference-interval laboratory states -> four-source observations.
CoDE's separate release constructs the frozen fact view, partitions, fitted models
and RQ analyses. Original Qwen3-4B generation and full-text matching are included.

No patient material, instantiated clinical prompts, model weights, ontology
embeddings, access credentials, institutional service addresses or original input
data are included. All runtime outputs named PRIVATE remain controlled clinical
artifacts. Run every production stage on the compute platform, not a local PC or
an SFTP/transfer server. This supplement was checked statically, not clinically rerun.

## Resources and environment

Use Linux/Bash and an existing accelerator-compatible Python 3.12/Torch environment.
Review requirements.txt and configs/environment_recorded.json; they do not install Torch. Install additional packages in
an isolated environment and retain the deployed Torch build. The historical reference
environment used Python 3.12.11, vendor Torch 2.8.0, transformers 4.57.6,
spaCy 3.8.2, medspaCy 1.3.1 and Pydantic 2.12.5. These are retained environment records,
not a newly verified joint pip lock. Fresh-install ranges can resolve differently;
preserve the original environment when reproducing its text boundaries. Some exact
pins were not recorded. Do not assume one universal requirements lock.

Obtain these resources through their applicable access routes:

- The authorized received source derivative: four *_hadm_info_first_diag.pkl files
  and its lab_test_mapping.csv. Source identity and file layout are documented below.
- EntityCoding NER/AC checkpoints at vendor/entity-coding/data/models/ner_model and
  ac_model. The exact upstream executable files and generic rules are vendored with
  their MIT notice; checkpoints are NOT included and have separate restrictions.
  The upstream release's download utility documents `data_download.py --models ner,ac`;
  obtain only those checkpoints under the upstream terms and place them in the paths
  above. This package does not execute upstream downloads or bundle the ICD model.
- HPO hp.obo, historical release 2026-09-01; recorded SHA-256 is
  93dace952fcb3ec4728818857f6ba76bc2d5312f4d83266519b8694b8e798f22.
- SapBERT derivative pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb,
  recorded revision ad34b857369884acb59a6e67f69008f533c8ca3e. Download all tokenizer
  and mean-pooling configuration files, not only the weights.
- The upstream RAG-HPO system_prompts.json from https://github.com/PoseyPod/RAG-HPO.
  The linker reads it to record prompt provenance; its actual adapted SYSTEM is
  fully included in scripts/rag_hpo_link_v2/link.py. Obtain the upstream file separately.
- Local Qwen/Qwen3.8-27B service. Historical download source was ModelScope;
  all 18 weight shards have recorded file revision
  1098534ab5d7220ea0f4a6b9f07bb03729a79c1d. Auxiliary files have their own revisions;
  this is not asserted as one immutable whole-repository/Hugging Face commit.
- Qwen/Qwen3-4B checkpoint and local service for optional original generation.
  Historical local file fingerprints are documented by CoDE; its download revision
  was not recovered. Model serving is external to these clients.

Copy configs/environment.example.sh to an untracked local file, configure paths and
TRACE_ENDPOINT, and set TRACE_ENV to it. Source generation, mapping and review use
the exact included prompts and validators; see docs/CONTRACTS.md and
configs/production_policy.json. The original configuration specified BF16 and no
weight quantization; API names and model config fields do not independently verify
the runtime dtype. Retain actual deployment records separately.

## Source-processing command order

Each invocation is foreground and exits when that stage completes or an error is
raised. Fresh prepare/build commands refuse existing outputs. Do not rerun a
prepare stage merely to resume an API run; resume its run command directly.

```bash
export TRACE_ENV="$PWD/configs/environment.local.sh"
bash run_source_pipeline.sh history-ner
bash run_source_pipeline.sh hpo-index
bash run_source_pipeline.sh hpo-link
bash run_source_pipeline.sh hpo-review
bash run_source_pipeline.sh ingest
bash run_source_pipeline.sh radiology-ner
bash run_source_pipeline.sh radiology-atoms
bash run_source_pipeline.sh radiology-bridge
bash run_source_pipeline.sh radiology-map-prepare
# Run the following on each of the eight mapping service nodes, using its own index.
bash run_source_pipeline.sh radiology-node 0
# ... nodes 1 through 7, all eight indices required ...
bash run_source_pipeline.sh radiology-export
bash run_source_pipeline.sh labs
bash run_source_pipeline.sh unified
```

Long stages can be launched with the compute platform's scheduler, or:

```bash
mkdir -p logs
nohup bash run_source_pipeline.sh hpo-index > logs/hpo-index.log 2>&1 < /dev/null &
echo $!
tail -f logs/hpo-index.log
```

NER and embedding commands require one eight-accelerator allocation. HPO review and
each radiology API shard default to 128 clients against the configured local service.
Retrieval uses 16 CPU threads; deterministic source/lab/unified exports use CPU/memory.
Past compute cost is not a guarantee for a different environment. Completion markers
are component reports/manifests and outputs/unified/acceptance_report.json. A stage
error requires inspection; do not discard failed records or change labels to proceed.
Ending the script does not release an external platform allocation.

The history/examination review exporter retains model flags and original links.
It can return exit code 2 after writing export_v1 when terminal review failures remain.
Inspect that acceptance file and verify complete original-mention coverage before
explicitly continuing; do not equate review failure with missing source observations.
Public wrappers do not auto-ignore this status. Original v1 retained 29 incomplete
review tasks rather than excluding their admissions.

The metadata helper reads the authorized lab mapping CSV, retaining only project
descriptions (not its count column). The historical inventory also held cohort
frequency columns; those are not used by these source/lab transformations. A fresh
metadata dictionary may therefore contain additional unused source items. The helper
does not recreate the historical inventory's byte identity or patient frequencies.

Audit partition membership is controlled, separate from diagnostic partitions, and
not published. Use REVIEW_SPLIT for the original authorized membership files, or the
wrapper's explicitly empty membership set for a fresh rebuild (export labels remaining).
Neither choice changes original mappings or applies review corrections.

## Output and CoDE boundary

outputs/unified contains source_documents_PRIVATE.jsonl, observations_PRIVATE.jsonl,
labels_PRIVATE.jsonl, admission_subject_map_PRIVATE.jsonl, patient_lab_view_PRIVATE.jsonl,
laboratory_derivations_PRIVATE.jsonl, the source/lab dictionaries, provenance and reports.
This is the observation export consumed by CoDE's frozen-view constructor, not an
already filtered model matrix. See docs/CONTRACTS.md for keys and state meanings.

Original F is the main-analysis view. The 85 source-fact actions are a separate
sensitivity and are not silently applied here. Review proposals remain separate.
Missing/null, explicit absence, family/hypothetical/possible assertions, conflicts and
unmapped concepts remain distinguishable. Radiology parent/atom links are not
independent clinical facts; the CoDE main view used original parent mappings.

The actual received source was a collaborator-provided derivative of MIMIC-CDM.
This code does not reconstruct or certify its exact relationship to the official
release, a universal pre-diagnosis cutoff, or authoritative subject identities.
The source adapter records inferred radiology-note-prefix subject groups and missing
times explicitly. Do not describe them as independently verified identities/times.
The original resource had 2,400 admissions, 4,800 history/examination documents,
5,960 received radiology reports, and 138,788 lab snapshot entries. The official
paper's 5,959 radiology reports and the received 5,960 remain a documented difference.

## Original rollout and final matching

FROZEN_INPUT is the separate authorized CoDE frozen rollout input with its manifest,
delivery contract, patient display facts, full 9,600 tasks, split without outcomes,
vocabulary and pilot membership. It is not distributed in this code-only supplement.

```bash
export FROZEN_INPUT=/authorized/frozen_rollout_input
bash run_original_rollouts.sh generate
bash run_original_rollouts.sh match-prepare
bash run_original_rollouts.sh match-node 0  # separately run indices 0 through 7
bash run_original_rollouts.sh match-merge
```

The original full-generation driver and final matching protocol are retained.
tools/prepare_match_source.py is a PUBLIC ADAPTER that reconstructs only the final
matcher's source envelope while preserving parsed explanation text and all visible F.
It does not reproduce historical mention/meaning inventories. Those intermediate
inventories did not feed final E(r); CoDE's final consumer uses whole-text fact matches.
For a historical source release, invoke matcher prepare with --source pointing to the
authorized original release instead. See docs/ORIGINAL_BINDING.md for retry/fallback
differences and the second-generator code boundary.

## Ownership and release status

This is a local code supplement candidate for integration into CoDE, not an upload
or a new license grant. Team-authored code licensing remains pending the authors'
decision, consistently with CoDE's LICENSE_PENDING.md. Preserve vendor licenses;
data/model/ontology terms are separate. See THIRD_PARTY_NOTICES.md.
