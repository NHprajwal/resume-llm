import json

INPUT_FILE = "training_data_normalized.jsonl"

REQUIRED_OUTPUT_KEYS = {
    "grammar",
    "skills",
    "experience",
    "projects",
    "overall_summary"
}

total = 0
valid = 0
invalid = 0
error_log = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        total += 1
        try:
            item = json.loads(line)

            # Validate input
            if "input" not in item:
                raise ValueError("Missing 'input' field")

            if not isinstance(item["input"], str) or not item["input"].strip():
                raise ValueError("'input' must be a non-empty string")

            # Validate output
            if "output" not in item:
                raise ValueError("Missing 'output' field")

            if not isinstance(item["output"], dict):
                raise ValueError("'output' must be a dictionary")

            missing_keys = REQUIRED_OUTPUT_KEYS - item["output"].keys()
            if missing_keys:
                raise ValueError(f"Missing output keys: {missing_keys}")

            valid += 1

        except Exception as e:
            invalid += 1
            error_log.append({
                "line": line_number,
                "error": str(e)
            })

print("\n========== DATASET VALIDATION REPORT ==========")
print(f"Total samples   : {total}")
print(f"Valid samples   : {valid}")
print(f"Invalid samples : {invalid}")

if error_log:
    print("\n❌ Sample Errors (first 10):")
    for err in error_log[:10]:
        print(f"Line {err['line']}: {err['error']}")

print("==============================================\n")
