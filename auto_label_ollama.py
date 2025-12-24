import json
import re
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI

# ---------------- CONFIG ----------------
INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "training_data_labeled.jsonl"

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_WORKERS = min(4, multiprocessing.cpu_count())

# 🔴 HARD-CODED API KEY (NOT RECOMMENDED)
client = OpenAI(
    api_key="sk-REPLACE_WITH_YOUR_REAL_KEY"
)

# ---------------------------------------------------------
# JSON extraction (robust fallback)
# ---------------------------------------------------------
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {"raw_output": text.strip()}

# ---------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------
def run_openai(prompt):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=400
    )
    return response.output_text.strip()

# ---------------------------------------------------------
# Worker (ONE sample)
# ---------------------------------------------------------
def process_one_entry(line):
    entry = json.loads(line)
    inp = entry["input"]

    prompt = f"""
{inp}

TASK:
Extract resume information.

OUTPUT RULES:
- Output ONLY valid minified JSON
- No explanations
- No extra text

JSON SCHEMA:
{{
  "grammar": "good|average|poor",
  "skills": [],
  "experience": [],
  "projects": [],
  "overall_summary": ""
}}
"""

    response = run_openai(prompt)
    entry["output"] = extract_json(response)
    return entry

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ training_data.jsonl not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"📦 Total samples: {total}")
    print(f"⚙️ Using {MAX_WORKERS} workers\n")

    # ✅ Create output file if not exists (append mode)
    open(OUTPUT_FILE, "a", encoding="utf-8").close()

    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one_entry, line) for line in lines]

        for future in as_completed(futures):
            try:
                result = future.result()

                # 🔒 Crash-safe write
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                completed += 1
                elapsed = time.time() - start_time
                avg = elapsed / completed
                remaining = total - completed
                eta = int(avg * remaining)

                print(
                    f"✔ Labeled {completed}/{total} | "
                    f"ETA {eta//60:02d}:{eta%60:02d}"
                )

            except Exception as e:
                print(f"❌ Failed entry: {e}")

    total_time = int(time.time() - start_time)
    print("\n✅ Auto-labeling completed!")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"📝 Total labeled: {completed}")
    print(f"⏱️ Time: {total_time//60:02d}:{total_time%60:02d}")

if __name__ == "__main__":
    main()
