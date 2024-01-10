#date: 2024-01-10T17:03:59Z
#url: https://api.github.com/gists/4fccfdd29c533b89d4cf9211b070bd31
#owner: https://api.github.com/users/roy-scalis

from lxml import etree, html

doc = html.parse(open('tos.html'))
elems = doc.xpath("//*")

for elem in elems:
    # drop all empty spans
    if elem.tag=='span' and len(elem.getchildren())<2 and (not elem.text or (elem.text and not elem.text.strip())):
        elem.drop_tag()
        continue

    elem.attrib.clear()

    # replace headings with divs
    if elem.tag in ['h1', 'h2', 'h3']:
        elem.attrib['class'] = elem.tag
        elem.tag = 'div'

htmlString = etree.tostring(doc.find('body'))
from bs4 import BeautifulSoup
print(BeautifulSoup(htmlString, 'html.parser').prettify())
