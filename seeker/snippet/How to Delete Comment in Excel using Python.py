#date: 2023-07-03T16:57:36Z
#url: https://api.github.com/gists/84dffb4a85191b5861c4f537d1ba9616
#owner: https://api.github.com/users/aspose-com-kb

import jpype
import asposecells
jpype.startJVM()
from asposecells.api import License, Workbook

# Instantiate the license
license = License()
license.setLicense("Aspose.Total.lic")

# Load Excel file
workbook = Workbook("InputWithComments.xlsx")

# Access a worksheet
worksheet = workbook.getWorksheets().get(0)

# Remove a specific comment from a cell
worksheet.getComments().removeAt("A3")

# Clear all the comments
# worksheet.ClearComments();

# Save output Excel file
workbook.save("NoComments.xlsx")
            
print("Comments removed successfully")

jpype.shutdownJVM()