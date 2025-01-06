#date: 2025-01-06T17:11:22Z
#url: https://api.github.com/gists/43bc6b4ac52e84f43cbdd5707896cd5a
#owner: https://api.github.com/users/KoStard

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.2.1",
#     "pillow>=11.1.0",
#     "pymupdf>=1.25.1",
# ]
# ///

import fitz  # PyMuPDF
from PIL import Image
import io
import sys

import numpy as np

def remove_green(image):
    """Convert green pixels to white in a PIL Image using numpy"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Separate RGB channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    # Create mask where green is dominant and above threshold
    green_mask = (g > r) & (g > b) & (g > 128)
    
    # Set green pixels to white
    img_array[green_mask] = [255, 255, 255]
    
    # Convert back to PIL Image
    return Image.fromarray(img_array)

def process_pdf(input_pdf, output_pdf):
    """Process PDF: convert pages to images, remove green, save as new PDF"""
    # Open the input PDF
    pdf_document = fitz.open(input_pdf)
    pdf_writer = fitz.open()  # For output PDF
    
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Convert page to image (300 DPI for good quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # Remove green highlights
        cleaned_img = remove_green(img)
        
        # Convert back to PDF page
        # Save cleaned image to a temporary bytes buffer in PNG format
        img_bytes = io.BytesIO()
        cleaned_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Create new PDF page and insert the image
        pdf_page = pdf_writer.new_page(width=cleaned_img.width, height=cleaned_img.height)
        pdf_page.insert_image(pdf_page.rect, stream=img_bytes)
    
    # Save the output PDF
    pdf_writer.save(output_pdf)
    pdf_writer.close()
    pdf_document.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_green_remover.py input.pdf output.pdf")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]
    
    process_pdf(input_pdf, output_pdf)
    print(f"Processed PDF saved to {output_pdf}")
