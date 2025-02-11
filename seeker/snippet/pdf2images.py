#date: 2025-02-11T17:09:11Z
#url: https://api.github.com/gists/d89df2209759bdf0e63f45389b451b59
#owner: https://api.github.com/users/zenoxygen

#!/usr/bin/env python3

#
# Usage: python pdf2images.py <pdf_path> [output_folder]
#

import argparse
import io
import os
import sys

import pymupdf
from PIL import Image


def extract_images_from_pdf(pdf_path, output_folder):
    try:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with pymupdf.open(pdf_path) as pdf_doc:
            try:
                os.makedirs(output_folder, exist_ok=True)
            except OSError as e:
                print(f"Error creating folder '{output_folder}': {e}")
                return

            print(f"Processing PDF: {pdf_path}")
            print(f"Output to: {output_folder}")

            for page_num in range(len(pdf_doc)):
                try:
                    page = pdf_doc.load_page(page_num)
                    img_list = page.get_images(full=True)
                except Exception as e:
                    print(f"Error loading page {page_num + 1}: {e}")
                    continue

                print(f"Page {page_num + 1}: Found {len(img_list)} images")

                for img_index, img in enumerate(img_list):
                    xref = img[0]
                    try:
                        base_image = pdf_doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image["ext"]
                        if not img_ext:
                            img_ext = "png"
                    except KeyError:
                        print(
                            f"Warning: Image {img_index + 1} page {page_num + 1} - incomplete data. Skipping."
                        )
                        continue
                    except Exception as e:
                        print(
                            f"Error extracting image {img_index + 1} page {page_num + 1}: {e}"
                        )
                        continue

                    try:
                        image = Image.open(io.BytesIO(img_bytes))
                    except Exception as e:
                        print(
                            f"Error opening image {img_index + 1} page {page_num + 1} with PIL: {e}"
                        )
                        continue

                    img_name = (
                        f"page_{page_num + 1}_img_{img_index + 1}.{img_ext.lower()}"
                    )
                    img_path = os.path.join(output_folder, img_name)

                    try:
                        image.save(img_path)
                        print(f"Saved: {img_path}")
                    except Exception as e:
                        print(f"Error saving {img_path}: {e}")

            print("Image extraction complete.")

    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from PDF.")
    parser.add_argument("pdf_path", help="Path to PDF file.")
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=".",
        help="Output folder (optional, defaults to current dir).",
    )
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)
    output_folder = os.path.abspath(args.output_folder)

    print(f"Input PDF: {pdf_path}")
    print(f"Output folder: {output_folder}")

    if not pdf_path.lower().endswith(".pdf"):
        print(f"Warning: '{pdf_path}' is not a pdf")

    try:
        with pymupdf.open(pdf_path) as pdf_test_open:
            pass
    except Exception as e:
        print(f"Error: cannot open PDF ({e})")
        sys.exit(1)

    extract_images_from_pdf(pdf_path, output_folder)


if __name__ == "__main__":
    main()
