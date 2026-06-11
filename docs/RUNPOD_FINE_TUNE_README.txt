NoLifeChatter RunPod fine-tuning pilot
======================================

Use this after double-clicking:

  4-export-finetune-pilot.bat

That creates:

  data\unsynced\fine_tune\persona_sft_runpod.zip

RunPod setup:

1. Go to https://www.runpod.io/
2. Pods -> Deploy
3. Secure Cloud
4. GPU: RTX 4090 24 GB
5. Template: official PyTorch / Jupyter
6. Storage: create/select a Network Volume in the same datacenter
7. Network Volume: 80 GB
8. Attach that Network Volume to the pod
9. Container Disk: leave default/20 GB
10. Deploy
11. Open JupyterLab
12. Upload persona_sft_runpod.zip into /workspace/

Pilot model:

  unsloth/Qwen2.5-7B-Instruct-bnb-4bit

This avoids Hugging Face gated-model login friction for the first paid run.

In RunPod terminal:

  cd /workspace
  python -m zipfile -e persona_sft_runpod.zip nlc_persona
  bash /workspace/nlc_persona/runpod_train_persona_lora.sh

When it finishes, download:

  /workspace/nlc_persona/persona_lora_result.zip

Then stop/terminate the pod so it does not keep billing.
