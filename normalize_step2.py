import json

INPUT_FILE = "training_data_labeled.jsonl"
OUTPUT_FILE = "training_data_normalized.jsonl"

DEFAULT_OUTPUT = {
    "grammar": "",
    "skills": [],
    "experience": {},
    "projects": {},
    "overall_summary": ""
}

fixed = 0
skipped = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        try:
            item = json.loads(line)

            # Skip only if input is invalid
            if "input" not in item or not isinstance(item["input"], str) or not item["input"].strip():
                skipped += 1
                continue

            output = item.get("output", {})

            normalized_output = {}

            for key, default_value in DEFAULT_OUTPUT.items():
                value = output.get(key, default_value)

                if value is None:
                    value = default_value

                # Skills must be list
                if key == "skills" and not isinstance(value, list):
                    value = [value] if isinstance(value, str) else []

                # Force dict for experience & projects
                if key in {"experience", "projects"} and not isinstance(value, dict):
                    value = {}

                # Force string fields
                if key in {"grammar", "overall_summary"} and not isinstance(value, str):
                    value = ""

                normalized_output[key] = value

            normalized_item = {
                "input": item["input"],
                "output": normalized_output
            }

            fout.write(json.dumps(normalized_item, ensure_ascii=False) + "\n")
            fixed += 1

        except Exception:
            skipped += 1

print("\n========== NORMALIZATION REPORT ==========")
print(f"Fixed & kept samples : {fixed}")
print(f"Skipped samples      : {skipped}")
print("=========================================\n")
