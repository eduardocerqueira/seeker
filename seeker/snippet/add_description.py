#date: 2025-12-31T16:41:18Z
#url: https://api.github.com/gists/cab1c1dbfd4d713a131b47f7bdc736b3
#owner: https://api.github.com/users/mitchkeller

import copy
import lxml.etree as ET
import sys

files = sys.argv[1:]

for file in files:
    tree = ET.parse(file)
    images = tree.xpath('//image[not(description)]|//interactive[@platform and not(description)]')

    # we'll loop through each <image>
    for image in images:

        # walk up the tree from the <image> to find a <figure>
        parent = image.getparent()
        while parent is not None:

            # if we find a <figure>, grab the caption, copy it, and add it as
            # a <description> element under the <image>
            if parent.tag == "figure":
                captions = parent.xpath('./caption')
                if len(captions) != 1:
                    print(f"Figure in {file} has zero or two captions")
                    break
                caption = captions[0]
                caption_cp = copy.deepcopy(caption)
                if caption_cp.text == None:
                    caption_cp.text = "ADD ALT TEXT TO THIS IMAGE"
                caption_cp.tag = "p"
                description = ET.Element("description")
                description.append(caption_cp)
                image.insert(0,description)
                break
            parent = parent.getparent()
        # if the image is not in a <figure>, cry out for help
        if parent is None:
            description = ET.SubElement(image, "description")
            default_text = ET.SubElement(description,"p")
            default_text.text = "ADD ALT TEXT TO THIS IMAGE"

    # Because PreTeXt provides default alt text referring to the image description when there is no shortdescription
    # we have agreed on the convention of relying on that default alt text. This removes any existing shortdescription
    # elements.
    shortdescs = tree.xpath('//shortdescription')

    for shortdesc in shortdescs:
        shortdesc.getparent().remove(shortdesc)

    tree.write(file, pretty_print=True)
