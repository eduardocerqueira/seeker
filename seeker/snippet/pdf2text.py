#date: 2023-07-07T17:05:56Z
#url: https://api.github.com/gists/64cd088e75492767ae05395c1679ae42
#owner: https://api.github.com/users/verheesj

import os
import PyPDF2
from pathlib import Path

# specify the directory you want to scan
input_directory = '/path/to/pdf/files'
output_directory = Path('/path/to/output/files')

if not output_directory.exists():
    output_directory.mkdir()

# loop through each file
for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        # form full file path
        file_path = os.path.join(input_directory, filename)
        pdfFileObj = open(file_path, 'rb')

        # initialize PDF file reader object
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # initialize a string to hold contents
        content = ""

        # iterate through each page and extract text
        for i in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(i)
            content += pageObj.extractText()

        # close the pdf file object
        pdfFileObj.close()

        # form the output file path
        output_file_path = output_directory / (filename.replace('.pdf', '.txt'))

        # write the content to the output file
        with open(output_file_path, 'w') as f:
            f.write(content)
