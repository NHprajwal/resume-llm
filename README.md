# 🧠 Resume Improvement LLM – End-to-End Fine-Tuning & Retraining Pipeline

This project builds a **custom LLM that parses resumes and suggests improvements** (grammar, skills, experience, projects) for a given role.

It supports:

* PDF / Image / DOCX resumes
* Automatic dataset generation
* Auto-labeling using a local LLM (Ollama)
* Cloud fine-tuning with LoRA (LLaMA / Mistral)
* **Incremental retraining on new resumes** (production-ready)

---

## 📁 Project Structure

```
fine-tuning-llm/
│
├── Data/
│   ├── WebDesigning/
│   ├── DataScience/
│   └── Backend/
│
├── extracted_text/
│   ├── *.txt
│   └── manifest.jsonl
│
├── training_data.jsonl
├── training_data_labeled.jsonl
│
├── resume_extractor.py
├── generate_training_data.py
├── auto_label_ollama.py
├── train.py
├── retrain.py
└── README.md
```

---

## 🔄 Full Pipeline Overview

```
Resume PDFs / Images / Docs
        ↓
resume_extractor.py
        ↓
extracted_text/*.txt + manifest.jsonl
        ↓
generate_training_data.py
        ↓
training_data.jsonl
        ↓
auto_label_ollama.py (local LLM)
        ↓
training_data_labeled.jsonl
        ↓
train.py (cloud GPU)
        ↓
Fine-tuned Resume LLM
```

---

## 1️⃣ resume_extractor.py

**Purpose:**

* Extract text from PDF / Image / DOCX resumes
* Create `manifest.jsonl`

**Output:**

```json
{
  "id": "ea82927f4c77c3f9",
  "role": "WebDesigning",
  "text": "Extracted resume text..."
}
```

---

## 2️⃣ generate_training_data.py

**Purpose:**

* Convert raw resume text into LLM-ready prompts

**Output (`training_data.jsonl`):**

```json
{
  "input": "Resume for Web Designer...",
  "output": ""
}
```

---

## 3️⃣ auto_label_ollama.py

**Purpose:**

* Auto-label resumes using a **local LLM** (Ollama)
* Generates structured suggestions

**Model options:**

* `llama3:8b-instruct`
* `phi3:instruct`
* `mistral:7b-instruct`

**Output (`training_data_labeled.jsonl`):**

```json
{
  "input": "Resume text...",
  "output": {
    "grammar": "Fix tense and punctuation",
    "skills": "Add React, Tailwind",
    "experience": "Quantify impact",
    "projects": "Clarify role",
    "overall_summary": "Good mid-level profile"
  }
}
```

---

## 4️⃣ train.py (Initial Fine-Tuning)

**Purpose:**

* Fine-tune base LLM using **LoRA (PEFT)**
* Runs on cloud GPU (RunPod / Colab / Paperspace)

**Base Models (recommended):**

* `meta-llama/Llama-3-8b-instruct`
* `mistralai/Mistral-7B-Instruct`

**Output:**

```
resume-model/
├── adapter_config.json
├── adapter_model.safetensors
```

This is **LoRA v1**.

---

# 🔁 Incremental Retraining Pipeline (IMPORTANT)

When you get **new resume PDFs**, you DO NOT retrain from scratch.

You **continue training from the previous LoRA adapter**.

---

## 5️⃣ retrain.py (Incremental Training)

### 🔹 Use this when new resumes arrive

### retrain.py

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Llama-3-8b-instruct"
OLD_ADAPTER = "resume-model"            # previous LoRA
NEW_DATA = "/workspace/data/training_data_new_labeled.jsonl"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto"
)

# Load previous adapter
model = PeftModel.from_pretrained(base_model, OLD_ADAPTER)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load new dataset
dataset = load_dataset("json", data_files=NEW_DATA, split="train")

def format_example(example):
    return f"""
### Resume:
{example['input']}

### Suggestions:
{example['output']}
"""

dataset = dataset.map(lambda x: {"text": format_example(x)})

training_args = TrainingArguments(
    output_dir="resume-model-v2",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    save_steps=500,
    logging_steps=20,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_args
)

trainer.train()
model.save_pretrained("resume-model-v2")
```

---

## 🔥 Retraining Strategy (Best Practice)

| Scenario           | Action             |
| ------------------ | ------------------ |
| New resumes weekly | Use `retrain.py`   |
| Dataset drift      | Mix old + new data |
| Model size grows   | Merge LoRA         |
| Major update       | Re-train from base |

---

## 🧩 Optional: Merge LoRA (Production)

```python
merged = model.merge_and_unload()
merged.save_pretrained("resume-model-merged")
```

---

## 🚀 Deployment Options

* Convert to GGUF → run with Ollama
* Wrap with FastAPI → Resume scoring API
* Integrate into ATS / HR tool

---

## ✅ Final Notes

* You now have a **production-grade LLM lifecycle**
* Fully offline labeling
* Cheap cloud training
* Continuous improvement

---

## 📬 Next Enhancements (Optional)

* Resume scoring (0–100)
* Role-specific prompts
* Grammar-only micro-model
* CI/CD retraining pipeline

---

👨‍💻 Built for scalable resume intelligence systems
