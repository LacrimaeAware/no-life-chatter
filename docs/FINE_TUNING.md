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

From the repo on the Windows machine:

```powershell
.\.venv\Scripts\python.exe scripts\export_persona_sft.py `
  --authors earnestsinceresugmamale,gero_30,forsenstares,ebbel `
  --max-examples-per-author 6000
```

Outputs:

- `data/unsynced/fine_tune/persona_train.jsonl`
- `data/unsynced/fine_tune/persona_val.jsonl`

These files contain real chat and are gitignored. Do not commit them.

For a broader first run:

```powershell
.\.venv\Scripts\python.exe scripts\export_persona_sft.py `
  --min-author-messages 1000 `
  --max-examples-per-author 8000
```

The exporter writes OpenAI-style chat JSONL:

- `system`: task instruction
- `user`: `<persona=name>` plus recent chat context
- `assistant`: the real next message that persona wrote

## GPU choice

Start with an 8B instruct model and LoRA/QLoRA.

- Pilot: RTX 4090 / RTX 3090 / A5000 class GPU, 24 GB VRAM.
- More comfortable/faster: A100 40/80 GB or H100.
- Base model target: Llama 3.1 8B Instruct or another 7B-9B model that runs
  well locally after quantization.

Do not start with a huge 70B model. The user's local RX 5700 XT is best suited
to running an 8B-ish quantized model in LM Studio, so a giant trained model
would be expensive to train and awkward to use.

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
