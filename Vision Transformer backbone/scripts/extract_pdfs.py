import os
from pypdf import PdfReader

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"Extracting: {pdf_path} -> {txt_path}")
    try:
        reader = PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                f.write(f"--- PAGE {i+1} ---\n")
                f.write(text)
                f.write("\n\n")
        print(f"Successfully extracted {len(reader.pages)} pages.")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "..", "paper", "references")
    for file in os.listdir(target_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(target_dir, file)
            txt_path = os.path.join(target_dir, file.replace(".pdf", "_extracted.txt"))
            extract_pdf_to_txt(pdf_path, txt_path)

if __name__ == "__main__":
    main()
