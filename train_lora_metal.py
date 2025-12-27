# ===============================
# Apple Silicon / Metal Safe Setup
# ===============================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TRAIN_FILE = "training_data_train.jsonl"
VAL_FILE = "training_data_val.jsonl"

OUTPUT_DIR = "./resume-lora"

MAX_LENGTH = 768        # Reduced for MPS safety
EPOCHS = 3
LR = 2e-4

# ===============================
# Tokenizer
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ===============================
# Load Dataset
# ===============================
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE
    }
)

def tokenize(example):
    text = ""
    for msg in example["messages"]:
        text += f"{msg['role'].upper()}: {msg['content']}\n"

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    remove_columns=["messages"],
    desc="Tokenizing dataset"
)

# ===============================
# Load Model (Metal Safe)
# ===============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map={"": "mps"},
    low_cpu_mem_usage=True,
    attn_implementation="eager"   # VERY IMPORTANT for Apple Silicon
)

# ===============================
# LoRA Configuration (Phi-3 correct modules)
# ===============================
lora_config = LoraConfig(
    r=4,                        # Reduced for memory safety
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["qkv_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# Training Arguments (Transformers ≥ 4.57)
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # Reduced for MPS
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    eval_strategy="steps",           # NEW API name
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# ===============================
# Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# ===============================
# Train
# ===============================
trainer.train()

# ===============================
# Save LoRA Adapter
# ===============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete. LoRA adapter saved to:", OUTPUT_DIR)
