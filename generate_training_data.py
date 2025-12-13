import os
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

MANIFEST_FILE = "extracted_texts/manifest.jsonl"
OUTPUT_FILE = "training_data.jsonl"

# ---------- CLEANING ----------

def clean_text(text):
    """Clean extracted OCR text to avoid garbage in training."""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?/()&%+\-\s]", "", text)
    return text.strip()


# ---------- TRAINING EXAMPLE BUILDER ----------

def build_training_pair(role, resume_text):
    """Build an LLM training pair with task-specific fields."""

    input_prompt = (
        f"You are an expert resume analyzer. "
        f"Here is a resume for the role '{role}'. "
        f"Analyze it and provide improvements.\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"Return improvements in structured JSON with keys: "
        f"'grammar', 'skills', 'experience', 'projects', 'overall_summary'."
    )

    output_placeholder = {
        "grammar": "",
        "skills": "",
        "experience": "",
        "projects": "",
        "overall_summary": ""
    }

    return {
        "input": input_prompt,
        "output": output_placeholder
    }


# ---------- WORKER FUNCTION ----------

def process_manifest_line(line):
    entry = json.loads(line)

    role = entry.get("role", "")
    text = clean_text(entry.get("text", ""))

    if len(text) < 100:
        return None

    return build_training_pair(role, text)


# ---------- MAIN (MULTIPROCESSING) ----------

def main():
    if not os.path.exists(MANIFEST_FILE):
        print("❌ manifest.jsonl not found. Run extraction script first.")
        return

    # Read all lines first
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📦 Total resumes found: {len(lines)}")

    dataset = []

    # Worker count (leave 2 cores free)
    workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"⚙️ Using {workers} parallel workers\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_manifest_line, line) for line in lines]

        for future in as_completed(futures):
            result = future.result()
            if result:
                dataset.append(result)

    # Write JSONL output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    print("\n✅ Training dataset created!")
    print(f"📁 Saved: {OUTPUT_FILE}")
    print(f"📝 Total examples: {len(dataset)}")


if __name__ == "__main__":
    main()
