#date: 2026-01-02T17:11:24Z
#url: https://api.github.com/gists/d5731c140454e4e58d97a0ed69b4a391
#owner: https://api.github.com/users/aspose-com-kb

import aspose.cells as cells

# Create a new workbook
workbook = cells.Workbook()

# Access the first worksheet
worksheet = workbook.worksheets.get(0)

# Input ml value
worksheet.cells.get("A1").put_value(100)  # 100 ml

# Convert ml to oz
ml_value = worksheet.cells.get("A1").double_value
oz_value = ml_value * 0.033814

# Output oz value
worksheet.cells.get("B1").put_value(oz_value)

# Save the workbook
workbook.save("MLtoOZConversion.xlsx")
