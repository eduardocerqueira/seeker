#date: 2024-10-30T17:01:56Z
#url: https://api.github.com/gists/575b101254d1c7c536970b7a1c6ab993
#owner: https://api.github.com/users/sskanishk

# Install Required Library
# pip install pymupdf

# --------------------------------------------

import fitz  # PyMuPDF

def highlight_text_in_pdf(pdf_path, output_path, targets):

    pdf_document = fitz.open(pdf_path)
    
    targets = [str(target) for target in targets]
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        for target in targets:
            areas = page.search_for(target)
            if areas:
                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.update()
                print(f"Highlighted: '{target}' on page {page_num + 1}")
            else:
                print(f"Target not found on page {page_num + 1}: '{target}'")
    
    pdf_document.save(output_path)
    pdf_document.close()
    print(f"Saved highlighted PDF as: {output_path}")

targets = ["123456", "Important Note", "RN789012", "2024"]

highlight_text_in_pdf("input_file.pdf", "output_highlighted_file.pdf", targets)


# -----------------------------------------------

# To run
# python/python3 highlight_text_in_pdf.py

