#date: 2022-05-26T17:16:10Z
#url: https://api.github.com/gists/67ac16beafe3eb54a0fe300bd0e7e7ff
#owner: https://api.github.com/users/evindunn

#!/usr/bin/env python3

import xml.etree.ElementTree as ET

from urllib.request import urlopen
from json import loads as json_loads
from xml.dom import minidom


def main():
    ipset = ET.Element("ipset", attrib={"type": "hash:net"})
    ipset_short = ET.SubElement(ipset, "short")
    ipset_short.text = "Google Cloud Platform IP ranges"

    with urlopen("https://www.gstatic.com/ipranges/cloud.json") as url:
        gcp_ipsets = json_loads(url.read().decode("utf-8"))

    for gcp_ipset in gcp_ipsets["prefixes"]:
        if "ipv4Prefix" not in gcp_ipset.keys():
            continue
       
        subnet = gcp_ipset["ipv4Prefix"]
        current_entry = ET.SubElement(ipset, "entry")
        current_entry.text = subnet


    dom = minidom.parseString(ET.tostring(ipset, encoding="unicode"))
    print(dom.toprettyxml())


if __name__ == "__main__":
    main()