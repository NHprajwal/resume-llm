import json
import subprocess
import re
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "training_data_labeled.jsonl"

MODEL = "llama3:8b"   # or: llama3:3b-instruct, mistral:7b-instruct

# ---------------------------------------------------------
# JSON extraction (robust)
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
# Run Ollama
# ---------------------------------------------------------
def run_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True
    )
    return result.stdout.strip()


# ---------------------------------------------------------
# Worker (ONE resume per process)
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

    response = run_ollama(prompt)
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
    print(f"📦 Total resumes: {total}")

    MAX_OLLAMA_WORKERS = 3
    workers = min(MAX_OLLAMA_WORKERS, multiprocessing.cpu_count())
    print(f"⚙️ Using {workers} Ollama workers\n")

    start_time = time.time()
    completed = 0
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_entry, line) for line in lines]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = total - completed
                eta_seconds = int(avg_time * remaining)

                eta_h = eta_seconds // 3600
                eta_m = (eta_seconds % 3600) // 60
                eta_s = eta_seconds % 60

                print(
                    f"✔ Labeled {completed} / {total} | "
                    f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

            except Exception as e:
                print(f"❌ Failed entry: {e}")

    # Write output JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_time = int(time.time() - start_time)
    h = total_time // 3600
    m = (total_time % 3600) // 60
    s = total_time % 60

    print("\n✅ Auto-labeling completed!")
    print(f"📁 Saved: {OUTPUT_FILE}")
    print(f"📝 Total labeled samples: {len(results)}")
    print(f"⏱️ Total time: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
