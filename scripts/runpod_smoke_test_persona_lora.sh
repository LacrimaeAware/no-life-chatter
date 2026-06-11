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
OUT="${2:-$(pwd)/persona_lora_smoke_test.txt}"

if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: LoRA adapter directory not found: $ADAPTER"
  echo "Expected the trained adapter at /workspace/nlc_persona/persona_lora"
  exit 1
fi

echo "== NoLifeChatter persona LoRA smoke test =="
echo "Adapter: $ADAPTER"
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

python smoke_test_persona_lora.py --adapter "$ADAPTER" --out "$OUT"

echo
echo "DONE. Open/download this file to inspect the outputs:"
echo "  $OUT"
