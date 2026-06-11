"""Train a persona LoRA on a rented NVIDIA GPU with Unsloth.

Run this on the GPU pod, not on the local Windows/AMD machine.

Expected inputs are the JSONL files created by scripts/export_persona_sft.py.
The script saves a LoRA adapter directory. Merging/quantizing to GGUF for LM
Studio is a separate step after the pilot looks good.
"""

from __future__ import annotations

import argparse
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
    from transformers import TrainingArguments
    from trl import SFTTrainer

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

    def format_batch(batch):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    data = data.map(format_batch, batched=True, remove_columns=data["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=250,
            save_total_limit=2,
            optim="adamw_8bit",
            fp16=True,
            bf16=False,
            report_to="none",
            seed=1337,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
