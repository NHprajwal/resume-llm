import json
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "training_data_labeled.jsonl"

MODEL = "llama3:8b"   # or: phi3:instruct, mistral:7b-instruct

# ---------------------------------------------------------
# JSON extraction (robust)
# ---------------------------------------------------------
def extract_json(text):
    """
    Extract JSON safely even if model prints extra text.
    """
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
# Worker (ONE resume per process call)
# ---------------------------------------------------------
def process_one_entry(line):
    entry = json.loads(line)
    inp = entry["input"]

    prompt = (
        inp +
        "\n\nRespond ONLY in JSON.\n"
        "Return a JSON object with keys: "
        "grammar, skills, experience, projects, overall_summary.\n"
    )

    print("🧠 Generating...")

    response = run_ollama(prompt)
    result_json = extract_json(response)

    entry["output"] = result_json
    return entry


# ---------------------------------------------------------
# MAIN (MULTIPROCESSING – LIMITED WORKERS)
# ---------------------------------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ training_data.jsonl not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📦 Total prompts: {len(lines)}")

    # 🚨 IMPORTANT: Limit Ollama workers
    MAX_OLLAMA_WORKERS = 3   # 🔥 DO NOT exceed 3 for llama3:8b
    workers = min(MAX_OLLAMA_WORKERS, multiprocessing.cpu_count())

    print(f"⚙️ Using {workers} Ollama workers\n")

    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_entry, line) for line in lines]

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print("✔ Done\n")
            except Exception as e:
                print(f"❌ Failed entry: {e}")

    # Write output JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n✅ Auto-labeling completed!")
    print(f"📁 Saved: {OUTPUT_FILE}")
    print(f"📝 Total labeled samples: {len(results)}")


if __name__ == "__main__":
    import os
    main()
