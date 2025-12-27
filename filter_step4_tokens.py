import json
from transformers import AutoTokenizer

INPUT_FILE = "training_data_chat.jsonl"
OUTPUT_FILE = "training_data_chat_filtered.jsonl"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_TOKENS = 4096

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

kept = 0
dropped = 0
max_seen = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        # Combine all message contents
        full_text = ""
        for msg in item["messages"]:
            full_text += msg["content"] + "\n"

        token_count = len(tokenizer(full_text)["input_ids"])
        max_seen = max(max_seen, token_count)

        if token_count <= MAX_TOKENS:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
        else:
            dropped += 1

print("\n========== TOKEN FILTER REPORT ==========")
print(f"Kept samples     : {kept}")
print(f"Dropped samples  : {dropped}")
print(f"Max tokens seen  : {max_seen}")
print(f"Token limit used : {MAX_TOKENS}")
print("========================================\n")
