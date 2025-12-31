#date: 2025-12-31T16:41:18Z
#url: https://api.github.com/gists/cab1c1dbfd4d713a131b47f7bdc736b3
#owner: https://api.github.com/users/mitchkeller

import lxml.etree as ET
import sys

files = sys.argv[1:]

for file in files:
  tree = ET.parse(file)
  images = tree.xpath('//image[not(description)]|//interactive[@platform and not(description)]')

  # we'll loop through each <image>
  for image in images:
    print(file + ": line " + str(image.sourceline))