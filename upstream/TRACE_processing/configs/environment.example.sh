# Source this file after replacing the authorized local paths.
export RAW_DIR="/authorized/source/mimic"
export ENTITY_REPO="$PWD/vendor/entity-coding"
export EMBED_MODEL="$PWD/models/SapBERT-mnli-snli-scinli-scitail-mednli-stsb"
export HPO_OBO="$PWD/resources/hp.obo"
export HPO_PROMPT_SOURCE="$PWD/resources/RAG-HPO/system_prompts.json"
export TRACE_ENDPOINT="http://localhost:8802/v1/chat/completions"
export GENERATION_ENDPOINT="http://localhost:8001/v1/chat/completions"
export QWEN4B_DIR="$PWD/models/Qwen3-4B"
export WORKERS=128
export PYTHON=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
