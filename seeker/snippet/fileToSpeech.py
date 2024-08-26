#date: 2024-08-26T17:09:58Z
#url: https://api.github.com/gists/e1ddf1766cee4e06a031299342c94fb2
#owner: https://api.github.com/users/alonsoir

import PyPDF2
import subprocess

# Open the PDF file (Enter Path To Your PDF)
file = open('fullnotes_lagrangiano_modelo_estandar.pdf', 'rb')
readpdf = PyPDF2.PdfReader(file)

# Iterate over each page in the PDF
for pagenumber in range(len(readpdf.pages)):
    # Extract text from the page
    page = readpdf.pages[pagenumber]
    text = page.extract_text()

    # Use the 'say' command to read the text
    subprocess.run(['say', '--progress', '-v', 'Flo', '-r', '140', text])

# Close the PDF file
file.close()