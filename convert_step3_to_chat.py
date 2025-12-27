import json

INPUT_FILE = "training_data_normalized.jsonl"
OUTPUT_FILE = "training_data_chat.jsonl"

SYSTEM_PROMPT = "You are an expert resume analyzer."

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        chat_item = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": item["input"]
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        item["output"],
                        ensure_ascii=False
                    )
                }
            ]
        }

        fout.write(json.dumps(chat_item) + "\n")

print("✅ Chat format conversion completed!")
