#!/usr/bin/env bash
set -euo pipefail

# Run this inside the RunPod terminal after unzipping persona_sft_runpod.zip:
#
#   cd /workspace
#   mkdir -p nlc_persona
#   unzip persona_sft_runpod.zip -d nlc_persona
#   bash /workspace/nlc_persona/runpod_train_persona_lora.sh

cd "$(dirname "$0")"

# Keep model/package caches on /workspace so the regular RunPod Volume Disk is
# the only disk that needs to be sized up for the pilot.
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export PIP_CACHE_DIR=/workspace/pip_cache
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

echo "== NoLifeChatter persona LoRA pilot =="
echo "Working directory: $(pwd)"
echo

if [ ! -f persona_train.jsonl ] || [ ! -f persona_val.jsonl ]; then
  echo "ERROR: persona_train.jsonl/persona_val.jsonl not found next to this script."
  echo "Make sure you unzipped persona_sft_runpod.zip into /workspace/nlc_persona."
  exit 1
fi

echo "[1/4] Creating Python environment..."
python -m venv /workspace/nlc_train_env
source /workspace/nlc_train_env/bin/activate
python -m pip install --upgrade pip

echo
echo "[2/4] Installing training libraries..."
python -m pip install unsloth datasets trl

echo
echo "[3/4] Training LoRA..."
python train_persona_lora_unsloth.py \
  --train "$(pwd)/persona_train.jsonl" \
  --val "$(pwd)/persona_val.jsonl" \
  --out "$(pwd)/persona_lora" \
  --epochs 1 \
  --rank 16

echo
echo "[4/4] Packaging result..."
zip -r persona_lora_result.zip persona_lora

echo
echo "DONE."
echo "Download this from RunPod Jupyter:"
echo "  $(pwd)/persona_lora_result.zip"
echo
echo "IMPORTANT: Stop/terminate the RunPod pod when you are finished downloading."
