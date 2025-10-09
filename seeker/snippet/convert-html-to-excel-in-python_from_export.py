#date: 2025-10-09T16:59:27Z
#url: https://api.github.com/gists/0c21e6a7d6169ea84e1f5cc5fb527aa2
#owner: https://api.github.com/users/aspose-com-gists

from aspose.cells import Workbook

workbook = Workbook("sample.html")

# Save to XLS format
workbook.save("output.xls")

# Save to CSV format
workbook.save("output.csv")

# Save to PDF for reporting
workbook.save("output.pdf")
