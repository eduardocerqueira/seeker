#date: 2024-09-04T16:53:25Z
#url: https://api.github.com/gists/fe3c9925ecddd9e89b7575c0446e642c
#owner: https://api.github.com/users/encryptize

#!/usr/bin/env python3
# Install required packages
# pip install "python-barcode[images]" fpdf2
#
# usage:
# python generate.py <start> <end>
# result:
# The script creates a pdf file of labels with the given barcode amount
# For example 100 labels starting from 10005500
#

import barcode
from barcode.writer import ImageWriter
from fpdf import FPDF
import io
import sys


# Generate ITF barcode and save to a BytesIO object (in memory)
def generate_itf_barcode(data):
    itf = barcode.get('itf', str(data), writer=ImageWriter())
    barcode_io = io.BytesIO()
    itf.write(barcode_io)
    barcode_io.seek(0)

    return barcode_io


def generate_pdf_with_barcodes(barcodes: tuple, output_pdf: str = "barcodes.pdf"):
    pdf = FPDF(format=(50.0, 30.0))
    for barcode in barcodes:
        barcode_img = generate_itf_barcode(barcode)

        # Add the image to the PDF
        pdf.add_page()
        pdf.image(barcode_img, keep_aspect_ratio=True, h=30, w=50, y=0,x=0)  

    # Save the generated PDF
    pdf.output(output_pdf)
    print(f"Barcodes saved to {output_pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: python {sys.argv[0]} <start> <amount>")
        exit(1)

    start = int(sys.argv[1])
    amount = int(sys.argv[2])

    generate_pdf_with_barcodes((x for x in range(start, start+amount)))
