#date: 2025-10-09T16:59:27Z
#url: https://api.github.com/gists/0c21e6a7d6169ea84e1f5cc5fb527aa2
#owner: https://api.github.com/users/aspose-com-gists

from aspose.cells import Workbook, HtmlLoadOptions
from io import BytesIO

# Step 1: Define HTML string
html_data = """
<table border='1'>
<tr><th>Product</th><th>Price</th><th>Quantity</th></tr>
<tr><td>Laptop</td><td>800</td><td>5</td></tr>
<tr><td>Phone</td><td>400</td><td>10</td></tr>
</table>
"""

# Step 2: Convert HTML string to bytes
html_bytes = BytesIO(html_data.encode('utf-8'))

# Step 3: Load HTML from memory
options = HtmlLoadOptions()
workbook = Workbook(html_bytes, options)

# Step 4: Save as Excel
workbook.save("from_string.xlsx")