#date: 2022-03-02T17:07:06Z
#url: https://api.github.com/gists/4e2f36fcdbdd8d4f46b5201ff6287987
#owner: https://api.github.com/users/aspose-com-kb

import jpype
import asposecells

# Start JVM
jpype.startJVM()
from asposecells.api import License, Workbook

# Load License
licenseHtmlToImage = License()
licenseHtmlToImage.setLicense("Aspose.Cells.lic")

# Create an instance of empty Workbook
workbook = Workbook()

# Get access to worksheets collection
worksheets = workbook.getWorksheets()

# Get first worksheet from the collection
worksheet = worksheets.get(0)

# Set values in different cells using Cells collection
worksheet.getCells().get("C1").setValue("Value in cell C1")
worksheet.getCells().get("D1").setValue("Value in cell D1")
worksheet.getCells().get("E1").setValue("Value in cell E1")

# Autofit the columns to display complete data in columns
worksheet.autoFitColumns()

# Save the output XLSX file
workbook.save("output.xlsx")

# Shutdown the JVM
jpype.shutdownJVM()