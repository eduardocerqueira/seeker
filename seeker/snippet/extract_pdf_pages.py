#date: 2025-02-26T16:59:04Z
#url: https://api.github.com/gists/9775ed1c7998476f705e5bbfc990c67b
#owner: https://api.github.com/users/bschne

import PyPDF2
from io import BytesIO

def extract_pdf_pages(pdf_path, first_page, last_page):
    """
    Extract specific pages from a PDF file.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        first_page (int): 1-based index of the first page to extract.
        last_page (int): 1-based index of the last page to extract.
    
    Returns:
        bytes: Bytes of a PDF file containing the extracted pages.
    """
    # Validate input parameters
    if not isinstance(first_page, int) or not isinstance(last_page, int):
        raise TypeError("Page numbers must be integers")
    
    if first_page < 1:
        raise ValueError("first_page must be at least 1")
    
    if first_page > last_page:
        raise ValueError("first_page cannot be greater than last_page")
    
    # Convert from 1-based to 0-based indexing
    first_page_idx = first_page - 1
    last_page_idx = last_page - 1
    
    # Create PDF reader
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Validate page ranges
        if last_page_idx >= len(reader.pages):
            raise ValueError(f"PDF only has {len(reader.pages)} pages, but last_page is {last_page}")
        
        # Create a new PDF writer
        writer = PyPDF2.PdfWriter()
        
        # Add the selected pages to the writer
        for page_num in range(first_page_idx, last_page_idx + 1):
            writer.add_page(reader.pages[page_num])
        
        # Write the output to a BytesIO object
        output = BytesIO()
        writer.write(output)
        
        # Return the bytes
        output.seek(0)
        return output.getvalue()