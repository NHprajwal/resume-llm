import os
import json
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ----- CHANGE THESE -----
DATA_DIR = "data"
OUTPUT_DIR = "extracted_texts"
POPPLER_PATH = r"C:\Users\GJ\Desktop\prajwal\fine tuning LLM\poppler-25.12.0\Library\bin"
# ------------------------

# Add Poppler to PATH
os.environ["PATH"] += os.pathsep + POPPLER_PATH

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

MANIFEST_FILE = os.path.join(OUTPUT_DIR, "manifest.jsonl")


# ---------------------------------------------------------
# PDF Extraction (including scanned image PDFs)
# ---------------------------------------------------------
def extract_from_pdf(pdf_path):
    """Extract text from normal PDFs + scanned image PDFs using OCR."""
    text = ""

    # Step 1: Try to extract text using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass

    # If pdfplumber extracted REAL text → return it
    if len(text.strip()) > 20:
        return text.strip()

    # Step 2: Fallback → Treat as image PDF and OCR it
    print(f"🔍 OCR fallback for scanned PDF: {pdf_path}")

    try:
        images = convert_from_path(pdf_path, dpi=300)
        ocr_text = ""

        for img in images:
            img = img.convert("RGB")  # avoid Tesseract errors
            ocr_text += pytesseract.image_to_string(img)

        return ocr_text.strip()

    except Exception as e:
        print(f"❌ OCR failed for {pdf_path}: {e}")
        return ""


# ---------------------------------------------------------
# Word (.docx) Extraction
# ---------------------------------------------------------
def extract_from_word(path):
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception:
        return ""


# ---------------------------------------------------------
# Image Extraction (JPG, PNG, TIFF)
# ---------------------------------------------------------
def extract_from_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


# ---------------------------------------------------------
# Auto-detect file type
# ---------------------------------------------------------
def extract_text(file_path):
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".pdf"):
        return extract_from_pdf(file_path)

    if file_path_lower.endswith(".docx"):
        return extract_from_word(file_path)

    if file_path_lower.endswith((".jpg", ".jpeg", ".png", ".tiff")):
        return extract_from_image(file_path)

    return ""


# ---------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------
def process_one_file(args):
    file_path, role = args
    extracted = extract_text(file_path)
    return file_path, role, extracted


# ---------------------------------------------------------
# Main (MULTIPROCESSING VERSION – FAST)
# ---------------------------------------------------------
def main():
    manifest_entries = []
    tasks = []

    # Collect file tasks
    for role in os.listdir(DATA_DIR):
        role_folder = os.path.join(DATA_DIR, role)
        if not os.path.isdir(role_folder):
            continue

        for file in os.listdir(role_folder):
            file_path = os.path.join(role_folder, file)
            tasks.append((file_path, role))

    print(f"📦 Total files found: {len(tasks)}")

    # Number of worker processes (CPU_count - 2)
    workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"⚙️ Using {workers} parallel workers\n")

    # Process all files in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_file, t): t for t in tasks}

        for future in as_completed(futures):
            file_path, role, extracted = future.result()

            print(f"📄 Done: {file_path}")

            if not extracted or len(extracted.strip()) < 20:
                print(f"⚠️ Could not extract: {file_path}")
                continue

            # Save extracted text file
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_filename = f"{role}_{base}.txt"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(extracted)

            # Add to manifest
            manifest_entries.append({
                "filename": out_filename,
                "role": role,
                "text": extracted
            })

    # Save manifest
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print("\n✅ Extraction Complete!")
    print(f"📂 Text files saved in: {OUTPUT_DIR}")
    print(f"📄 Manifest saved as: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
