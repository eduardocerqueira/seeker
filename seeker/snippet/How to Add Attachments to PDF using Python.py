#date: 2023-02-16T16:38:04Z
#url: https://api.github.com/gists/10f5e97b36e3894e49ae542471e4a368
#owner: https://api.github.com/users/aspose-com-kb

import aspose.pdf as pdf
# Load License
license = pdf.License()
license.set_license("Aspose.Total.lic")

# Open document
pdfDocument =  pdf.Document("Input.pdf")

# Setup the FileSpecification object
fileSpecification = pdf.FileSpecification("input.png", "Sample Image File")

# Add attachment to the PDF
pdfDocument.embedded_files.add("1",fileSpecification)

# Save the document
pdfDocument.save("AddAttachment.pdf")

print("Attachment added successfully in PDF")