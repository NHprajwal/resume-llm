import pdfplumber

with pdfplumber.open("test data/Resume_H_C_Prajwal-6.pdf") as pdf:
    for page in pdf.pages:
        print("Extracted:", page.extract_text())
