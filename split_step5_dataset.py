import json
import random

INPUT_FILE = "training_data_chat_filtered.jsonl"

TRAIN_FILE = "training_data_train.jsonl"
VAL_FILE = "training_data_val.jsonl"
TEST_FILE = "training_data_test.jsonl"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

# For reproducibility
random.seed(42)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

total = len(lines)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_data = lines[:train_end]
val_data = lines[train_end:val_end]
test_data = lines[val_end:]

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_data)

with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.writelines(val_data)

with open(TEST_FILE, "w", encoding="utf-8") as f:
    f.writelines(test_data)

print("\n========== DATASET SPLIT REPORT ==========")
print(f"Total samples : {total}")
print(f"Train samples : {len(train_data)}")
print(f"Val samples   : {len(val_data)}")
print(f"Test samples  : {len(test_data)}")
print("=========================================\n")
