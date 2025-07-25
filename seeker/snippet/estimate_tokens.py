#date: 2025-07-25T17:16:55Z
#url: https://api.github.com/gists/aa7df8160ec1e37a296463d890ecf184
#owner: https://api.github.com/users/briantwalter

#!/usr/bin/env python3

import os
import glob
import fitz  # PyMuPDF
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and folder path from environment
API_KEY = os.getenv("VOYAGE_API_KEY")
PDF_FOLDER = os.getenv("PDF_FOLDER")

# Validate environment variables
if not API_KEY or not PDF_FOLDER:
    raise ValueError("Missing VOYAGE_API_KEY or PDF_FOLDER in .env file.")

# Initialize Voyage client
voyage = voyageai.Client(api_key=API_KEY)

def extract_text_from_pdf(filepath):
    """Extract and sanitize plain text from a PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        print(f"⚠️  Warning while processing {os.path.basename(filepath)}: {e}")

    if not isinstance(text, str):
        return ""

    # Clean up text: remove nulls, enforce UTF-8, strip whitespace
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace("\x00", "").strip()

    return text

 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"s "**********"t "**********"i "**********"m "**********"a "**********"t "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"i "**********"n "**********"_ "**********"f "**********"o "**********"l "**********"d "**********"e "**********"r "**********"( "**********"f "**********"o "**********"l "**********"d "**********"e "**********"r "**********"_ "**********"p "**********"a "**********"t "**********"h "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"n "**********"a "**********"m "**********"e "**********"= "**********"" "**********"v "**********"o "**********"y "**********"a "**********"g "**********"e "**********"- "**********"3 "**********". "**********"5 "**********"" "**********") "**********": "**********"
    """Estimates total token usage for all PDFs in the specified folder."""
    total_tokens = "**********"
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDF files found in: {folder_path}")
        return 0

    for file_path in pdf_files:
        text = extract_text_from_pdf(file_path)

        if not text or len(text.strip()) == 0:
            print(f"⚠️ Skipping {file_path} — no extractable text.")
            continue

        try:
            # The Voyage API expects a list of strings, not a single string
            token_count = "**********"=model_name)
            print(f"{os.path.basename(file_path)}: "**********"
            total_tokens += "**********"
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    print(f"\n✅ Total estimated tokens across all PDFs: "**********"
    return total_tokens

# 🔧 Run script
estimate_tokens_in_folder(PDF_FOLDER)
