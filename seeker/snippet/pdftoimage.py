#date: 2023-01-18T16:44:21Z
#url: https://api.github.com/gists/6cbe007ee44908ee98bf84bda8b2efd4
#owner: https://api.github.com/users/yadavmanoj354

from PIL import Image

# Open the PDF file
with open("file.pdf", "rb") as f:
    pdf = Image.open(f)

# Iterate over all pages
for i in range(0, pdf.get_page_count()):
    # Set the current page
    pdf.seek(i)
    # Save the current page as a JPEG file
    pdf.save("page_{}.jpg".format(i), "JPEG")
