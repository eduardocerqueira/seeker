#date: 2023-05-19T17:03:10Z
#url: https://api.github.com/gists/dc6af65e6b88c3e5d653d8b32641513e
#owner: https://api.github.com/users/aspose-com-kb

import jpype
jpype.startJVM()

from asposecells.api import Workbook

# Load Excel file
workbook = Workbook("test.xlsx")

# Access a worksheet
worksheet = workbook.getWorksheets().get(0)

# Insert two rows at 2nd position
worksheet.getCells().insertRows(1,2)

# Save output Excel file
workbook.save("insertedRows.xlsx")

jpype.shutdownJVM()