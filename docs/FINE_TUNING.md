# Persona fine-tuning runbook

Goal: train a local open-model LoRA so personas learn chat voice from the
archive, then run the result locally in LM Studio. This is separate from RAG:
fine-tuning teaches style and habits; RAG retrieves exact memories.

## Recommended order

1. Run a cheap pilot on the current archive.
2. Compare it against current RAG personas.
3. Ingest older logs if the pilot is promising.
4. Re-export and train the durable run.

Older logs are useful, but not mandatory. They mostly help with rare users,
old bits, and long-running references. Current logs are enough to prove whether
the training pipeline improves voice.

## Local export

Preferred path: double-click this file on the Windows machine:

```text
4-export-finetune-pilot.bat
```

Outputs:

- `data/unsynced/fine_tune/persona_train.jsonl`
- `data/unsynced/fine_tune/persona_val.jsonl`
- `data/unsynced/fine_tune/persona_sft_runpod.zip`

These files contain real chat and are gitignored. Do not commit them.

The batch file exports all regular chatters in the configured archive/channels:
minimum 1,000 messages and maximum 6,000 training examples per author. That is
the recommended first pilot. It is broader than the old four-person example and
better matches the goal of modeling the whole chat.

Manual equivalent, only if the batch file fails:

```powershell
.\.venv\Scripts\python.exe scripts\export_persona_sft.py `
  --min-author-messages 1000 `
  --max-examples-per-author 8000
```

The exporter writes OpenAI-style chat JSONL:

- `system`: task instruction
- `user`: `<persona=name>` plus recent chat context
- `assistant`: the real next message that persona wrote

For the first paid pilot, use this exact Windows sequence:

```text
Double-click 4-export-finetune-pilot.bat
```

Then upload `data\unsynced\fine_tune\persona_sft_runpod.zip` to the RunPod Jupyter
file browser, into `/workspace/`.

In the RunPod terminal, the only commands you should need are:

```bash
cd /workspace
mkdir -p nlc_persona
unzip persona_sft_runpod.zip -d nlc_persona
bash /workspace/nlc_persona/runpod_train_persona_lora.sh
```

When it finishes, download `/workspace/nlc_persona/persona_lora_result.zip`.
Then stop/terminate the pod.

## GPU choice

Start with an 8B instruct model and LoRA/QLoRA.

- Pilot: RTX 4090 / RTX 3090 / A5000 class GPU, 24 GB VRAM.
- More comfortable/faster: A100 40/80 GB or H100.
- Base model target: Llama 3.1 8B Instruct or another 7B-9B model that runs
  well locally after quantization.

Do not start with a huge 70B model. The user's local RX 5700 XT is best suited
to running an 8B-ish quantized model in LM Studio, so a giant trained model
would be expensive to train and awkward to use.

## Exact pilot rental

Use this for the first paid run:

1. Go to `https://www.runpod.io/`.
2. Create/log into an account and add about `$25` credit. Do not add `$100`
   for the pilot.
3. Go to Pods -> Deploy.
4. Choose **Secure Cloud**.
5. Choose **RTX 4090, 24 GB VRAM**.
6. Choose the official **PyTorch / Jupyter** template.
7. Create/select a **Network Volume** in the same datacenter as the GPU.
8. Set the Network Volume to `80 GB`.
9. Attach that Network Volume to the pod.
10. Leave **Container Disk** at the default, usually `20 GB`.
11. Deploy the pod.
12. Open JupyterLab or Web Terminal.

Why this exact choice: RTX 4090 has enough VRAM for an 8B QLoRA run, is much
cheaper than A100/H100, and trains the same kind of LoRA that can later run
locally after merge/quantization.

Disk note: RunPod has three similarly named storage choices. For this workflow,
use a **Network Volume** because it mounts at `/workspace`, survives pod
termination, and is cheaper than leaving a stopped pod's regular Volume Disk
around. The helper script keeps Hugging Face, datasets, pip cache, temp files,
checkpoints, and the LoRA output under `/workspace`, so attaching a Network
Volume makes the run more restart-safe. `80 GB` is boring and safe. Container
Disk is just the temporary container/session area, so the default is fine here.

Important: a Network Volume is tied to a datacenter. Create it in the same
datacenter where you can actually launch the RTX 4090/A5000/L40S pod. If that
datacenter has no GPU availability, either pick a GPU in that datacenter or
delete/recreate the Network Volume in another one.

If no RTX 4090 is available, pick this fallback order:

1. RTX 3090 24 GB
2. RTX A5000 24 GB
3. L40S 48 GB
4. A100 40 GB

Do not use CPU-only pods. Stop/terminate the pod when training is done; storage
and idle runtime can keep costing money.

## Training shape

Use LoRA or QLoRA, not full fine-tuning.

Suggested first hyperparameters:

- epochs: 1 to 2
- LoRA rank: 16 or 32
- learning rate: around `2e-4` for LoRA/QLoRA
- context length: 2048 or 4096
- validation split: exporter default 5%

Evaluate before spending more. If the model gets worse or starts overfitting,
lower epochs or clean the export filters.

## Safety and output

Local training means provider moderation is not the bottleneck, but Twitch is
still the output boundary. The bot must keep doing:

```text
generate -> output_filter.is_clean -> send
```

The blocklist file stays private in `data/unsynced/blocklist.txt`.

## General knowledge and archive questions

Fine-tuning is not the best way to answer questions like:

> do we have an emote of the bottle dog?

That should be a separate routed feature:

1. Retrieve relevant archive rows and emote names for "bottle dog".
2. Feed those retrieved facts to a stronger general LLM or local model.
3. Answer with cited/retrieved chat evidence.

That is archive-Q&A RAG, not persona fine-tuning. The two systems can share the
same SQLite archive but should stay separate.
