#date: 2024-02-16T17:04:55Z
#url: https://api.github.com/gists/9022cf8b54730b3ed4742a98e1847383
#owner: https://api.github.com/users/dsavchenko

import xml.etree.ElementTree as ET


def CDATA(text=None):
    element = ET.Element('![CDATA[')
    element.text = text
    return element

ET._original_serialize_xml = ET._serialize_xml


def _serialize_xml(write, elem, encoding, qnames, namespaces):
    if elem.tag == '![CDATA[':
        write("<%s%s]]>%s" % (elem.tag, elem.text, elem.tail))
        return
    return ET._original_serialize_xml(
        write, elem, encoding, qnames, namespaces)
ET._serialize_xml = ET._serialize['xml'] = _serialize_xml