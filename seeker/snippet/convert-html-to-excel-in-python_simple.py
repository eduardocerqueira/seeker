#date: 2025-10-09T16:59:27Z
#url: https://api.github.com/gists/0c21e6a7d6169ea84e1f5cc5fb527aa2
#owner: https://api.github.com/users/aspose-com-gists

from aspose.cells import Workbook

# Step 1: Define the input HTML file path
input_file = "sample.html"

# Step 2: Create a Workbook object and load the HTML
workbook = Workbook(input_file)

# Step 3: Save the file as Excel
workbook.save("output.xlsx")