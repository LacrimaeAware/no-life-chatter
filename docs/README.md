# NoLifeChatter Docs

This is the active documentation index. If a doc is dated, superseded, or mainly
chronological, it belongs in `docs/archive/` so it does not compete with the
current operating plan.

## Read First

- [STATE_OF_OPERATION.md](STATE_OF_OPERATION.md) - current runtime state,
  artifact status, safety boundaries, and return-to-project checklist.
- [ROADMAP.md](ROADMAP.md) - current state of the art, ranked remaining work,
  and the recommended next implementation order.
- [COMMANDS.md](COMMANDS.md) - live command bible. This is checked against
  `commands/*.py` by `scripts/audit_commands.py`.

## Active Reference

- [CHAT_ARCHIVE.md](CHAT_ARCHIVE.md) - archive schema, Chatterino log format,
  import/search design, and context-window caveats.
- [CHAT_PERSONALITY_RESEARCH.md](CHAT_PERSONALITY_RESEARCH.md) - research notes
  for embeddings, axes, emotes, social maps, and persona evaluation.
- [GENERATE_AND_BOT_MODES.md](GENERATE_AND_BOT_MODES.md) - generation commands,
  live resident persona controls, and remaining volition/queue work.
- [FINE_TUNING.md](FINE_TUNING.md) - LoRA/export/runbook notes. Current
  priority is still eval/RAG/memory before another blind training run.
- [IDEA_BANK.md](IDEA_BANK.md) - parked ideas that are not part of the immediate
  build queue.
- [RUNPOD_FINE_TUNE_README.txt](RUNPOD_FINE_TUNE_README.txt) - compact RunPod
  helper note kept for old fine-tune runs.

## Historical Archive

Historical handoffs, dated audits, and superseded roadmaps live in
[archive/](archive/). They are useful provenance, not the current agenda.

## Freshness Checks

Use this before trusting docs or generated persona artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\freshness_check.py
```

The freshness check wraps command import/doc coverage, README command coverage,
generated artifact status, docs layout sanity, latest rebuild-log status, and
git dirtiness.
