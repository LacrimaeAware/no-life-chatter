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

The pilot launcher (kept privately in `_private/finetune/`, since it names real
channels and chatters) exports a curated single-channel pilot: one channel,
selected high-value chatters, known same-person aliases merged, bot accounts
excluded, and maximum 5,000 training examples per author. That is the
recommended first real pilot: focused enough to improve the people you care
about, but still small enough to run on a single RTX 4090.

Manual equivalent (substitute your own channel/users):

```powershell
.\.venv\Scripts\python.exe scripts\export_persona_sft.py `
  --channels yourchannel `
  --authors user1,user2,user3 `
  --user-aliases altaccount=user1 `
  --exclude-users nightbot,yourbotaccount `
  --max-examples-per-author 5000
```

The exporter writes OpenAI-style chat JSONL:

- `system`: task instruction
- `user`: `<persona=name>` plus recent chat context
- `assistant`: the real next message that persona wrote

Training is ordinary supervised fine-tuning / next-token prediction on those
assistant messages. With current TRL versions the script requests
completion-only loss, so the model is mainly penalized for predicting the
persona's reply, not for recreating the prompt text. It does **not** train the
retrieval database, and it does not memorize or index facts the way RAG does.
RAG remains the memory system; LoRA is only a style/voice prior.

For the first paid pilot, use this exact Windows sequence:

```text
Double-click 4-export-finetune-pilot.bat
```

Then upload `data\unsynced\fine_tune\persona_sft_runpod.zip` to the RunPod Jupyter
file browser, into `/workspace/`.

In the RunPod terminal, the only commands you should need are:

```bash
cd /workspace
rm -rf nlc_persona
python -m zipfile -e persona_sft_runpod.zip nlc_persona
bash nlc_persona/RUN_ME_ON_RUNPOD.sh
```

When it finishes, download `/workspace/nlc_persona/persona_lora_result.zip`.
Then stop/terminate the pod.

Back on Windows, double-click:

```text
7-install-runpod-lora-result.bat
```

That copies the downloaded zip from `Downloads` into the gitignored private
fine-tune folder and extracts it to:

```text
data\unsynced\fine_tune\persona_lora_result
```

This result is a **LoRA adapter**, not a standalone LM Studio GGUF yet. Next
steps after install are:

1. Quick smoke-test the adapter against a few prompts.
2. If it looks promising, merge the adapter with its base model.
3. Convert/quantize the merged model to GGUF for LM Studio or run it through a
   server that supports PEFT/LoRA adapters directly.
4. Compare LoRA-only, RAG-only, and LoRA+RAG bot replies.

If the RunPod pod or its Network Volume is still available, smoke-test the
adapter there before converting. On Windows, double-click:

```text
8-copy-runpod-smoke-test-command.bat
```

Then paste the copied command into the RunPod terminal. It fetches
`scripts/smoke_test_persona_lora.py` and `scripts/runpod_smoke_test_persona_lora.sh`,
loads `/workspace/nlc_persona/persona_lora`, and writes:

```text
/workspace/nlc_persona/persona_lora_smoke_test.txt
```

The smoke test includes direct `@persona`-style prompts, normal and hyper-style
instructions, old bot outputs for comparison, and two samples per prompt. Open
or download that text file and inspect the sample outputs. If they look
meaningfully better than the base/RAG-only behavior, proceed to merge/convert.
If they look bad, fix the prompt/eval shape, dataset/export, or base model
choice before spending time on GGUF conversion.

Current pilot result as of 2026-06-11: training completed and the adapter was
installed locally under the private fine-tune folder, but the LoRA-only smoke
test looked mixed/bland. This is not ready to merge into LM Studio as the live
persona model yet. The current hypothesis is that LoRA-only is the wrong
evaluation surface: the intended product is **LoRA + RAG**, while the first
smoke test used only the adapter and tiny hand-written context windows.

To compare the RunPod LoRA smoke-test text against the current local LM Studio
RAG behavior, put the downloaded RunPod file here:

```text
data\unsynced\fine_tune\persona_lora_smoke_test.txt
```

Then double-click:

```text
9-compare-lora-vs-local-rag.bat
```

This writes a private report:

```text
data\unsynced\fine_tune\persona_lora_vs_local_rag.md
```

That report is **RAG-only on the local model** versus **LoRA-only on RunPod**.
It still does not test LoRA+RAG together. The next real evaluation should run
the trained adapter on RunPod with prompts that include the same author-only RAG
exemplars used by the live bot, or merge/serve the adapter somewhere safe and
run the existing persona prompt against it.

For that missing LoRA+RAG test, use:

```text
10-export-lora-rag-smoke-cases.bat
```

That creates:

```text
data\unsynced\fine_tune\persona_lora_rag_smoke_cases.json
```

Upload that JSON to:

```text
/workspace/nlc_persona/persona_lora_rag_smoke_cases.json
```

Then double-click:

```text
11-copy-runpod-lora-rag-smoke-command.bat
```

Paste the copied command into the RunPod terminal. It fetches the LoRA+RAG
smoke-test runner, loads `/workspace/nlc_persona/persona_lora`, and writes:

```text
/workspace/nlc_persona/persona_lora_rag_smoke_test.txt
```

This still does not touch the live bot. It is only an offline evaluation of the
trained Qwen LoRA adapter with the bot's RAG evidence included in the prompt.

If training is interrupted, do **not** delete `nlc_persona` before resuming.
The training script saves checkpoints under `/workspace/nlc_persona/persona_lora`
every 100 optimizer steps and automatically resumes from the newest
`checkpoint-*` directory. To resume after an interruption:

```bash
cd /workspace/nlc_persona
bash RUN_ME_ON_RUNPOD.sh
```

If the scripts were updated after the zip was extracted, refresh only the
scripts and keep the dataset/checkpoints:

```bash
cd /workspace/nlc_persona && python - <<'PY'
from urllib.request import urlopen
base = "https://raw.githubusercontent.com/LacrimaeAware/no-life-chatter/main/scripts/"
for name in ["train_persona_lora_unsloth.py", "runpod_train_persona_lora.sh", "RUN_ME_ON_RUNPOD.sh"]:
    print("fetching", name)
    open(name, "wb").write(urlopen(base + name, timeout=60).read())
PY
bash RUN_ME_ON_RUNPOD.sh
```

## GPU choice

Start with an 8B instruct model and LoRA/QLoRA.

- Pilot: RTX 4090 / RTX 3090 / A5000 class GPU, 24 GB VRAM.
- More comfortable/faster: A100 40/80 GB or H100.
- Pilot base model target: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`.
  This was chosen as a low-friction pipeline pilot because it is ungated and
  easy to train through Unsloth, not because it is guaranteed to be the best
  persona-voice model. If the pilot is technically successful but feels bland,
  rerun on the model family the user actually wants to run long-term, e.g.
  Llama 3.1/3.2 8B Instruct or another strong 7B-9B chat model.
- Later comparison target: Llama 3.1/3.2 8B Instruct or another 7B-9B model
  that runs well locally after quantization.

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

Why this exact choice: RTX 4090 has enough VRAM for a 7B/8B QLoRA run, is much
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

Current pilot hyperparameters:

- epochs: 1
- LoRA rank: 16
- learning rate: `2e-4`
- context length: 2048
- effective batch size: 16 (`batch_size=2`, `grad_accum=8`)
- validation split: exporter default 5%
- checkpoint/save interval: every 100 optimizer steps

Observed pilot shape on an RTX 4090:

- 41,278 train examples
- 2,186 validation examples
- 2,580 optimizer steps
- prompt+completion training, so loss is focused on persona replies
- bf16 precision on RTX 4090
- observed speed at step 191: about 2.36 seconds/step including one eval

Expected duration for this exact pilot is roughly 1.5 to 2.25 hours after setup
and tokenization. Eval/checkpoint overhead makes exact ETA wobble; the progress
bar after the first 50 to 100 steps is the best source of truth.

Evaluate before spending more. If the model gets worse or starts overfitting,
lower epochs or clean the export filters.

## RunPod troubleshooting notes

- `unzip: command not found`: use `python -m zipfile -e ...` as documented.
- `evaluation_strategy` / `eval_strategy` errors: fixed in
  `scripts/train_persona_lora_unsloth.py` by checking the installed
  Transformers/TRL signatures.
- `fp16` vs `bf16` Unsloth error: fixed by detecting CUDA bf16 support and using
  bf16 on RTX 4090.
- `.nfs... Device or resource busy` during tokenization: caused by temporary
  multiprocessing files on RunPod Network Volume. The current wrapper puts
  temp files under `/tmp/nlc_train_tmp` and caps TRL dataset preprocessing to
  one worker when supported.
- `Ctrl+C` / `KeyboardInterrupt`: training stops. If it happened before the
  first checkpoint save, progress is lost. With current scripts, reruns resume
  from the newest checkpoint once at least step 100 has saved.

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
