#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export PIP_CACHE_DIR=/workspace/pip_cache
export TMPDIR=/tmp/nlc_train_tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

ADAPTER="${1:-$(pwd)/persona_lora}"
CASES="${2:-$(pwd)/persona_lora_rag_smoke_cases.json}"
OUT="${3:-$(pwd)/persona_lora_rag_smoke_test.txt}"

if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: LoRA adapter directory not found: $ADAPTER"
  echo "Expected the trained adapter at /workspace/nlc_persona/persona_lora"
  exit 1
fi

if [ ! -f "$CASES" ]; then
  echo "ERROR: RAG smoke cases not found: $CASES"
  echo "Run 10-export-lora-rag-smoke-cases.bat locally, then upload"
  echo "data/unsynced/fine_tune/persona_lora_rag_smoke_cases.json"
  echo "to /workspace/nlc_persona/ on RunPod."
  exit 1
fi

echo "== NoLifeChatter LoRA + RAG smoke test =="
echo "Adapter: $ADAPTER"
echo "Cases:   $CASES"
echo "Output:  $OUT"
echo

if [ -f /workspace/nlc_train_env/bin/activate ]; then
  source /workspace/nlc_train_env/bin/activate
else
  echo "Training venv not found; creating a small test environment."
  python -m venv /workspace/nlc_train_env
  source /workspace/nlc_train_env/bin/activate
  python -m pip install --upgrade pip
  python -m pip install unsloth transformers peft accelerate bitsandbytes
fi

python smoke_test_persona_lora_with_rag.py --adapter "$ADAPTER" --cases "$CASES" --out "$OUT"

echo
echo "DONE. Open/download this file to inspect the LoRA+RAG outputs:"
echo "  $OUT"
