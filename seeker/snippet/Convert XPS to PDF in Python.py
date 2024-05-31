#date: 2024-05-31T17:09:27Z
#url: https://api.github.com/gists/28fb5a38c566daef19e8c87e8eb97d78
#owner: https://api.github.com/users/aspose-com-kb

import aspose.page
from aspose.page import * 
from io import BytesIO

# Load input XPS file
document = aspose.page.xps.XpsDocument("input.xps")

# Initiate PdfSaveOptions class object
options = aspose.page.xps.presentation.pdf.PdfSaveOptions()

# Create Stream for the PDF file 
ms = BytesIO()

# Initiate PdfDevice object
device = aspose.page.xps.presentation.pdf.PdfDevice(ms)

# Convert XPS to PDF 
document.save(device, options)

# Export the output file
with open("output.pdf","wb") as file:
    file.write(ms.getbuffer())
