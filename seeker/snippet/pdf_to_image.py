#date: 2025-07-18T17:00:14Z
#url: https://api.github.com/gists/8be2059a01889ee620f58f16771d3db7
#owner: https://api.github.com/users/uberdeveloper

from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        pages = convert_from_path(pdf_path)
        num_pages = len(pages)
        if num_pages > 10:
            print("Too many pages")
            return
        # Create output folder based on filename
        output_folder = os.path.splitext(os.path.basename(pdf_path))[0]
        os.makedirs(output_folder, exist_ok=True)

        for i, page in enumerate(pages):
            filename = os.path.join(output_folder, f"page_{i+1}.png")
            page.save(filename, 'PNG')
            print(f"Saved {filename}")
    except Exception as e:
        print(f"Error: {e}")

pdf_file = "1_Bella_Casa_Fashion___Retail_Ltd_539399.pdf"
pdf_to_images(pdf_file)
