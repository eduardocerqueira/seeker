#date: 2022-08-26T16:44:32Z
#url: https://api.github.com/gists/cfb68bcee4677951d8c10b1d3c38b8a6
#owner: https://api.github.com/users/bleso-a

from trp import Document
doc = Document(response)
page_string = ''
for page in doc.pages:
    for line in page.lines:
        page_string += " "
        page_string += str(line.text)
print(page_string)