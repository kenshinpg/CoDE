#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source "${TRACE_ENV:-configs/environment.example.sh}"
STAGE="${1:?Choose history-ner, hpo-index, hpo-link, hpo-review, ingest, radiology-ner, radiology-atoms, radiology-bridge, radiology-map-prepare, radiology-node, radiology-export, labs, unified}"
shift
mkdir -p outputs logs
case "$STAGE" in
history-ner)
  "$PYTHON" scripts/entitycoding_v1/prepare.py --data "$RAW_DIR" --out outputs/history_documents --per-disease 0
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/entitycoding_v1/run.py --repo "$ENTITY_REPO" --input outputs/history_documents/documents.jsonl --out outputs/history_examination/extraction --ac-batch-size 16
  "$PYTHON" scripts/entitycoding_v1/accept.py --run outputs/history_examination/extraction ;;
hpo-index)
  "$PYTHON" scripts/hpo_review_v2/prepare_all_mapping.py --entity-dir outputs/history_examination/extraction --obo "$HPO_OBO" --model "$EMBED_MODEL" --out outputs/hpo/frozen
  "$PYTHON" scripts/rag_hpo_native_fix_v1/fix.py init --source outputs/hpo/frozen --model "$EMBED_MODEL" --out outputs/hpo/index
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/rag_hpo_native_fix_v1/fix.py embed --run outputs/hpo/index
  "$PYTHON" scripts/rag_hpo_native_fix_v1/fix.py gate --run outputs/hpo/index
  "$PYTHON" scripts/rag_hpo_native_fix_v1/fix.py retrieve --run outputs/hpo/index ;;
hpo-link)
  "$PYTHON" scripts/rag_hpo_link_v2/link.py --candidates outputs/hpo/index/candidates.jsonl --out outputs/hpo/link --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS" --prompt-source "$HPO_PROMPT_SOURCE" --max-attempts 3 ;;
hpo-review)
  "$PYTHON" scripts/hpo_review_full_v1/review.py prepare --link outputs/hpo/link --entity outputs/history_examination/extraction --patients 2400 --map-batch 16 --out outputs/hpo/review
  "$PYTHON" scripts/hpo_review_full_v1/review.py run --run outputs/hpo/review --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS"
  "$PYTHON" scripts/hpo_review_full_v1/review.py run --run outputs/hpo/review --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS" --retry-failed
  "$PYTHON" tools/prepare_local_resources.py review-partitions --out outputs/review_membership
  # Original private audit membership can be supplied instead. It is not a diagnostic split.
  "$PYTHON" scripts/hpo_review_full_v1/export.py --run outputs/hpo/review --split "${REVIEW_SPLIT:-outputs/review_membership}" ;;
ingest)
  "$PYTHON" tools/prepare_local_resources.py lab-dictionary --mapping-csv "$RAW_DIR/lab_test_mapping.csv" --out outputs/resources/lab_inventory.json
  "$PYTHON" scripts/trace2400_ingest_v1/prepare.py --raw-dir "$RAW_DIR" --entity-dir outputs/history_examination/extraction --reviewed outputs/hpo/review/export_v1/mentions_reviewed_PRIVATE.jsonl --lab-inventory outputs/resources/lab_inventory.json --out outputs/source --contract scripts/trace2400_unified_v1/code_contracts.py ;;
radiology-ner)
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/entitycoding_v1/run.py --repo "$ENTITY_REPO" --input outputs/source/radiology_entity_input_PRIVATE.jsonl --out outputs/radiology_extraction --ac-batch-size 16
  "$PYTHON" scripts/entitycoding_v1/accept.py --run outputs/radiology_extraction ;;
radiology-atoms)
  cat outputs/radiology_extraction/node{0..7}.jsonl > outputs/radiology_entity_documents_PRIVATE.jsonl
  "$PYTHON" scripts/radiology_atoms_v1/atoms.py prepare --input outputs/radiology_entity_documents_PRIVATE.jsonl --documents 0 --batch 8 --out outputs/radiology_atoms
  "$PYTHON" scripts/radiology_atoms_v1/atoms.py run --run outputs/radiology_atoms --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS" ;;
radiology-bridge)
  "$PYTHON" scripts/radiology_bridge_v1/bridge.py build --run outputs/radiology_atoms --out outputs/radiology_bridge --atom-code scripts/radiology_atoms_v1/atoms.py ;;
radiology-map-prepare)
  "$PYTHON" scripts/radiology_mapping_v1/pipeline.py prepare --bridge outputs/radiology_bridge --index outputs/hpo/index --model "$EMBED_MODEL" --cpu-threads 16 --out outputs/radiology_mapping ;;
radiology-node)
  NODE="${1:?Pass shard 0..7 on its local Qwen service node}"
  "$PYTHON" scripts/radiology_mapping_v1/pipeline.py run --root outputs/radiology_mapping --stage map --shard "$NODE" --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS"
  "$PYTHON" scripts/radiology_mapping_v1/pipeline.py run --root outputs/radiology_mapping --stage review --shard "$NODE" --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B --workers "$WORKERS" ;;
radiology-export)
  "$PYTHON" scripts/radiology_mapping_v1/accept.py --root outputs/radiology_mapping --out outputs/radiology_export ;;
labs)
  "$PYTHON" scripts/trace2400_lab_discrete_v2/build.py --source outputs/source --out outputs/laboratory --unit-policy source_paired_assumption ;;
unified)
  "$PYTHON" scripts/trace2400_unified_v1/build.py --source outputs/source --radiology outputs/radiology_export --lab outputs/laboratory --out outputs/unified ;;
*) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
