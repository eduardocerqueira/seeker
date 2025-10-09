#date: 2025-10-09T16:59:27Z
#url: https://api.github.com/gists/0c21e6a7d6169ea84e1f5cc5fb527aa2
#owner: https://api.github.com/users/aspose-com-gists

from aspose.cells import Workbook, HtmlLoadOptions

# Step 1: Set HTML load options
load_options = HtmlLoadOptions()
load_options.auto_fit_cols_and_rows = True  # Automatically adjusts columns and rows

# Step 2: Load HTML with options
workbook = Workbook("sample.html", load_options)

# Step 3: Save as Excel
workbook.save("table_advanced.xlsx")