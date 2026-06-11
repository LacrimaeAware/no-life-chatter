"""Train a persona LoRA on a rented NVIDIA GPU with Unsloth.

Run this on the GPU pod, not on the local Windows/AMD machine.

Expected inputs are the JSONL files created by scripts/export_persona_sft.py.
The script saves a LoRA adapter directory. Merging/quantizing to GGUF for LM
Studio is a separate step after the pilot looks good.
"""

from __future__ import annotations

import argparse
import inspect
import importlib.metadata as metadata
import math
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="/workspace/nlc_persona/persona_train.jsonl")
    ap.add_argument("--val", default="/workspace/nlc_persona/persona_val.jsonl")
    ap.add_argument("--out", default="/workspace/nlc_persona/persona_lora")
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except Exception:
        SFTConfig = None

    def package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except Exception:
            return "unknown"

    print(
        "Versions: "
        f"torch={torch.__version__}, "
        f"unsloth={package_version('unsloth')}, "
        f"transformers={package_version('transformers')}, "
        f"trl={package_version('trl')}, "
        f"datasets={package_version('datasets')}",
        flush=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1337,
    )

    data = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
    )

    def prompt_completion_batch(batch):
        prompts, completions = [], []
        for messages in batch["messages"]:
            prompts.append(messages[:-1])
            completions.append([messages[-1]])
        return {"prompt": prompts, "completion": completions}

    data = data.map(
        prompt_completion_batch,
        batched=True,
        remove_columns=data["train"].column_names,
    )
    effective_batch = max(1, args.batch_size * args.grad_accum)
    steps_per_epoch = math.ceil(len(data["train"]) / effective_batch)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    warmup_steps = max(1, int(total_steps * 0.03))
    print(
        "Training plan: "
        f"{len(data['train']):,} train examples, {len(data['validation']):,} validation examples, "
        f"effective batch {effective_batch}, about {total_steps:,} optimizer steps. "
        f"precision {'bf16' if use_bf16 else 'fp16'}, warmup {warmup_steps} steps. "
        "Logs every 10 steps, eval every 100 steps, save every 250 steps.",
        flush=True,
    )

    training_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_first_step=True,
        eval_steps=100,
        save_steps=250,
        save_total_limit=2,
        optim="adamw_8bit",
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="none",
        seed=1337,
        disable_tqdm=False,
    )
    args_cls = SFTConfig or TrainingArguments
    args_params = inspect.signature(args_cls).parameters
    if "eval_strategy" in args_params:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in args_params:
        training_kwargs["evaluation_strategy"] = "steps"
    if SFTConfig is not None:
        if "max_length" in args_params:
            training_kwargs["max_length"] = args.max_seq_length
        if "packing" in args_params:
            training_kwargs["packing"] = False
        if "completion_only_loss" in args_params:
            training_kwargs["completion_only_loss"] = True
        if "eos_token" in args_params:
            training_kwargs["eos_token"] = "<|im_end|>"
    training_args = args_cls(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        args=training_args,
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
