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
6. Container Disk: 80 GB
7. Network Volume: leave empty/off for the pilot
8. Deploy
9. Open JupyterLab
10. Upload persona_sft_runpod.zip into /workspace/

In RunPod terminal:

  cd /workspace
  mkdir -p nlc_persona
  unzip persona_sft_runpod.zip -d nlc_persona
  bash /workspace/nlc_persona/runpod_train_persona_lora.sh

When it finishes, download:

  /workspace/nlc_persona/persona_lora_result.zip

Then stop/terminate the pod so it does not keep billing.
