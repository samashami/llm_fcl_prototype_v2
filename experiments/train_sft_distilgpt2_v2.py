# experiments/train_sft_distilgpt2_v2.py
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


class SFTJsonlDataset(Dataset):
    """
    Simple PyTorch dataset for our SFT JSONL:
      each line: {"prompt": "...", "response": "...", ...}

    We concatenate prompt + response and train the LM to predict the whole sequence.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = []
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                response = obj.get("response", "")
                text = prompt + response
                self.examples.append(text)

        if len(self.examples) == 0:
            raise ValueError(f"No examples found in {jsonl_path}")

        print(f"[SFTJsonlDataset] Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # Trainer expects tensors, not lists
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # For causal LM, labels are just the input IDs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def parse_args():
    ap = argparse.ArgumentParser(description="SFT DistilGPT2 on controller traces (JSONL v2)")
    ap.add_argument(
        "--jsonl",
        type=str,
        default="datasets/sft/sft_pairs_v2.jsonl",
        help="Path to SFT JSONL data",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="models/sft_distilgpt2_v2",
        help="Where to save fine-tuned model",
    )
    ap.add_argument(
        "--num_train_epochs",
        type=int,
        default=12,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading tokenizer and base model (distilgpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        # GPT2-like models have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Build dataset (no HuggingFace datasets / pyarrow)
    train_dataset = SFTJsonlDataset(jsonl_path, tokenizer, max_length=args.max_length)

    # Training setup
    # Training setup
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to=[],  # disable wandb etc.
        fp16=False,    # CPU training on Mac

        use_mps_device=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"[train] Starting training on {len(train_dataset)} examples...")
    trainer.train()

    print(f"[train] Saving fine-tuned model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()