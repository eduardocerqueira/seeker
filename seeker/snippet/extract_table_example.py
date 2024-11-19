#date: 2024-11-19T17:00:33Z
#url: https://api.github.com/gists/9203619987005e067afcdb5ca695a99f
#owner: https://api.github.com/users/sdg-1

import pdfplumber
import PyPDF2

# Open the PDF file with PyPDF2

pdf_file = 'example.pdf'
pdf_reader = PyPDF2.PdfReader(pdf_file)

# Extract text with PyPDF2
full_text = ""

for page in pdf_reader.pages:
    full_text += page.extract_text()  # Extract text from each page

print("Extracted Text:", full_text)

# Extract tables with pdfplumber
with pdfplumber.open(pdf_file) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()  # Extract tables
        for table in tables:
            print("Extracted Table:", table)  # Print each table