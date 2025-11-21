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


# ------------ CONFIG ---------------

BASE_DIR = Path(__file__).resolve().parent.parent
SFT_PATH = BASE_DIR / "datasets" / "sft" / "sft_pairs_v2.jsonl"

MODEL_NAME = "distilgpt2"  # starting base model
OUTPUT_DIR = BASE_DIR / "models" / "sft_distilgpt2_v2"

MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-5


# ------------ DATASET ---------------

class SFTJsonlDataset(Dataset):
    """
    Simple dataset: each line is {"prompt": ..., "response": ...}
    We train on prompt + response as a single sequence.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                response = obj["response"]
                text = prompt + "\n\n" + response
                self.examples.append(text)

        print(f"Loaded {len(self.examples)} SFT examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # causal LM: labels = input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# ------------ MAIN TRAINING ---------------

def main():
    print(f"Using SFT dataset: {SFT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 family sometimes has no pad token -> set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    dataset = SFTJsonlDataset(SFT_PATH, tokenizer, max_length=MAX_LENGTH)

    # small train/val split (90/10)
    n = len(dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        weight_decay=0.0,
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()