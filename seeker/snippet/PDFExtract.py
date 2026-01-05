#date: 2026-01-05T17:01:41Z
#url: https://api.github.com/gists/600a2d4dfa8338d0a3b5203445ecc128
#owner: https://api.github.com/users/adrievx

import fitz
import os

pdf_path = "input.pdf"
output_folder = "extracted_images"

os.makedirs(output_folder, exist_ok=True)

doc = fitz.open(pdf_path)

for page_index in range(len(doc)):
    for img_index, img in enumerate(doc.get_page_images(page_index)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        with open(f"{output_folder}/page{page_index+1}_img{img_index+1}.{image_ext}", "wb") as f:
            f.write(image_bytes)
