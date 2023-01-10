#date: 2023-01-10T16:59:17Z
#url: https://api.github.com/gists/6f13d4cf4b34af7fa18a4287c55405c2
#owner: https://api.github.com/users/aspose-com-kb

import aspose.pdf as pdf

# Load the license
license = pdf.License()
license.set_license("Aspose.Total.lic")

# Create PdfFileEditor object
pdfFileEditor = pdf.facades.PdfFileEditor()

# Create an empty array
pdffiles = []
# Add target PDF file names
pdffiles.append('FirstFile.pdf')
pdffiles.append('SecondFile.pdf')
pdffiles.append('ThirdFile.pdf')

pdfFileEditor.concatenate(pdffiles, "MergedFile.pdf")

print("Merging process completed successfully")