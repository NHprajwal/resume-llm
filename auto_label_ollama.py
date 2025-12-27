import json
import subprocess
import re
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "training_data_labeled.jsonl"

MODEL = "phi3:instruct"   # 🔥 Best for Mac M4 (Metal GPU)

# ---------------------------------------------------------
# Resume compression (token reduction)
# ---------------------------------------------------------
def compress_resume(text, max_chars=2500):
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


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
# Run Ollama (Metal GPU auto-enabled)
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
# Worker (single resume)
# ---------------------------------------------------------
def process_one_entry(line):
    entry = json.loads(line)

    resume_text = compress_resume(entry["input"])

    prompt = f"""
Analyze the resume below.
Return ONLY valid JSON with keys:
grammar, skills, experience, projects, overall_summary.
Keep each field under 120 words.

Resume:
{resume_text}
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
    print(f"\n📦 Total resumes: {total}")

    # ✅ Mac M4 safe worker count
    TOTAL_RAM_GB = 32   # adjust if needed
    MAX_WORKERS = 4 if TOTAL_RAM_GB >= 32 else 2
    workers = min(MAX_WORKERS, multiprocessing.cpu_count())

    print(f"⚙️ Using {workers} Ollama workers (Metal GPU enabled)\n")

    results = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_entry, line) for line in lines]

        for future in as_completed(futures):
            try:
                results.append(future.result())
                completed += 1

                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                remaining = total - completed
                eta = remaining / speed if speed > 0 else 0

                print(
                    f"✔ {completed}/{total} | "
                    f"{speed:.2f} resumes/sec | "
                    f"ETA: {eta/60:.1f} min"
                )

            except Exception as e:
                print(f"❌ Failed entry: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time
    print("\n✅ Auto-labeling completed!")
    print(f"📁 Saved: {OUTPUT_FILE}")
    print(f"📝 Total labeled samples: {len(results)}")
    print(f"⏱ Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Avg speed: {len(results)/total_time:.2f} resumes/sec")


if __name__ == "__main__":
    main()
