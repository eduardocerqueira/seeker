#date: 2021-12-15T17:17:36Z
#url: https://api.github.com/gists/617928df406026b5fd382fc845cf0fc3
#owner: https://api.github.com/users/tomasonjo

import requests
import pdf2image
import pytesseract

pdf_link = "https://www.dailyscript.com/scripts/the_matrix.pdf"

pdf = requests.get(pdf_link)
doc = pdf2image.convert_from_bytes(pdf.content)

# Get the article text
article = []
for page_number, page_data in enumerate(doc):
    # First page is the title
    if page_number == 0:
      continue
    txt = pytesseract.image_to_string(page_data, lang='eng').encode("utf-8")
    article.append(txt.decode("utf-8"))
article_txt = " ".join(article)