#!/usr/bin/env bash
set -euo pipefail

# Run this inside the RunPod terminal after extracting persona_sft_runpod.zip:
#
#   cd /workspace
#   rm -rf nlc_persona
#   python -m zipfile -e persona_sft_runpod.zip nlc_persona
#   bash nlc_persona/RUN_ME_ON_RUNPOD.sh

cd "$(dirname "$0")"

# Keep model/package caches on /workspace so the regular RunPod Volume Disk is
# the only disk that needs to be sized up for the pilot.
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_datasets_cache
export PIP_CACHE_DIR=/workspace/pip_cache
# Keep short-lived multiprocessing temp files off the RunPod Network Volume.
# Network volumes can leave .nfs* cleanup files behind while worker processes
# still hold descriptors; local /tmp avoids that noisy failure mode.
export TMPDIR=/tmp/nlc_train_tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

echo "== NoLifeChatter persona LoRA pilot =="
echo "Working directory: $(pwd)"
echo

if [ ! -f persona_train.jsonl ] || [ ! -f persona_val.jsonl ]; then
  echo "ERROR: persona_train.jsonl/persona_val.jsonl not found next to this script."
  echo "Make sure you extracted persona_sft_runpod.zip into /workspace/nlc_persona."
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
# v2 default: an ABLITERATED base so the persona isn't fighting refusal
# alignment (the pilot's plain Qwen kept refusing edgy jokes). Override by
# exporting MODEL / EOS_TOKEN before running. EOS MUST match the base family
# (Llama-3.x: <|eot_id|>, Qwen2.5: <|im_end|>) or completion masking breaks.
# If the primary 401s on download, try a fallback:
#   MODEL=huihui-ai/Llama-3.1-8B-Instruct-abliterated  EOS_TOKEN='<|eot_id|>'
#   MODEL=huihui-ai/Qwen2.5-7B-Instruct-abliterated     EOS_TOKEN='<|im_end|>'
MODEL="${MODEL:-mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated}"
EOS_TOKEN="${EOS_TOKEN:-<|eot_id|>}"
echo "Base model: $MODEL   EOS: $EOS_TOKEN"
python train_persona_lora_unsloth.py \
  --train "$(pwd)/persona_train.jsonl" \
  --val "$(pwd)/persona_val.jsonl" \
  --out "$(pwd)/persona_lora" \
  --model "$MODEL" \
  --eos-token "$EOS_TOKEN" \
  --epochs 1 \
  --rank 16

echo
echo "[4/4] Packaging result..."
python - <<'PY'
from pathlib import Path
import shutil

target = Path("persona_lora_result.zip")
if target.exists():
    target.unlink()
shutil.make_archive("persona_lora_result", "zip", "persona_lora")
PY

echo
echo "DONE."
echo "Download this from RunPod Jupyter:"
echo "  $(pwd)/persona_lora_result.zip"
echo
echo "IMPORTANT: Stop/terminate the RunPod pod when you are finished downloading."
