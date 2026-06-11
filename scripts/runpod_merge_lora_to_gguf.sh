#!/usr/bin/env bash
set -euo pipefail
# Merge a trained persona LoRA into its base model and export a GGUF that
# LM Studio can load directly. This is the step that makes a trained adapter
# *usable* — LM Studio loads GGUF models, not raw PEFT adapters.
#
# Run on a RunPod pod (ideally the one that trained it — the base is cached).
# Output: persona_merged_Q4_K_M.gguf  → download it, drop it in LM Studio's
# models folder, load it, and the bot's [llm] endpoint now speaks through the
# fine-tune. Then ~persona / the eval (scripts/eval_personas.py) use it for free.
#
# Override via env: ADAPTER=... BASE=... OUT=...

cd "$(dirname "$0")"
ADAPTER="${ADAPTER:-/workspace/nlc_persona/persona_lora}"
BASE="${BASE:-mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated}"
OUT="${OUT:-/workspace/nlc_persona/persona_merged}"
QUANT="${QUANT:-Q4_K_M}"

source /workspace/nlc_train_env/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/hf_cache
pip install -q -U peft transformers accelerate sentencepiece gguf
# transformers imports torchvision for vision models; on a torch-version-mismatched
# pod that import crashes (circular import) and breaks text conversion too. We
# don't need vision here — remove it so transformers skips that import.
pip uninstall -y torchvision >/dev/null 2>&1 || true

# Locate the adapter (folder containing adapter_config.json). Extract the
# result zip if only that exists.
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  if [ -f /workspace/nlc_persona/persona_lora_result.zip ]; then
    echo "Adapter folder missing — extracting persona_lora_result.zip..."
    rm -rf /workspace/nlc_persona/_adapter && \
      python -m zipfile -e /workspace/nlc_persona/persona_lora_result.zip /workspace/nlc_persona/_adapter
  fi
  found="$(find /workspace/nlc_persona -maxdepth 4 -name adapter_config.json 2>/dev/null | head -1)"
  [ -n "$found" ] && ADAPTER="$(dirname "$found")"
fi
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "ERROR: couldn't find adapter_config.json anywhere under /workspace/nlc_persona."
  echo "Make sure your trained adapter (or persona_lora_result.zip) is there."
  exit 1
fi
echo "Using adapter: $ADAPTER"

echo "[1/3] Merging adapter into base (this is the LoRA -> plain model step)..."
python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    "$BASE", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "$ADAPTER")
model = model.merge_and_unload()           # bake the LoRA into the weights
model.save_pretrained("$OUT", safe_serialization=True)
AutoTokenizer.from_pretrained("$BASE").save_pretrained("$OUT")
print("merged ->", "$OUT")
PY

# Newer transformers saves a tokenizer_class (e.g. "TokenizersBackend") that
# llama.cpp's converter can't instantiate — normalize it to the fast class.
python - <<PY
import json, os
p = os.path.join("$OUT", "tokenizer_config.json")
try:
    d = json.load(open(p))
    if "Backend" in str(d.get("tokenizer_class", "")) or not d.get("tokenizer_class"):
        d["tokenizer_class"] = "PreTrainedTokenizerFast"
        json.dump(d, open(p, "w"))
        print("normalized tokenizer_class for gguf conversion")
except Exception as e:
    print("tokenizer_config fix skipped:", e)
PY

echo "[2/3] Converting merged model to GGUF (f16)..."
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py "$OUT" \
  --outfile persona_merged_f16.gguf --outtype f16

echo "[3/3] Quantizing to $QUANT (fits 8 GB VRAM for LM Studio)..."
command -v cmake >/dev/null || { apt-get update -y >/dev/null && apt-get install -y cmake build-essential >/dev/null; }
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF >/dev/null
cmake --build llama.cpp/build --target llama-quantize -j >/dev/null
./llama.cpp/build/bin/llama-quantize \
  persona_merged_f16.gguf "persona_merged_${QUANT}.gguf" "$QUANT"

echo
echo "DONE -> $(pwd)/persona_merged_${QUANT}.gguf"
echo "Download it, put it in LM Studio's models folder, load it, and the bot"
echo "is now running the fine-tuned weights (RAG prompt unchanged). Stop the pod."
