#date: 2022-06-10T16:59:00Z
#url: https://api.github.com/gists/2850b2528b93db342f263bc34fd5bcf5
#owner: https://api.github.com/users/bauripalash

from xml.etree import ElementTree




feeds= """<PASTE_OPML_HERE>"""


print("---")
tree = ElementTree.fromstring(feeds)


for node in tree.iter("outline"):
    if "htmlUrl" in node.attrib and "text" in node.attrib:
        print("* [{}][{}]({})".format(node.attrib["text"],node.attrib["description"],node.attrib["htmlUrl"]))
  




#print(feeds)
