#date: 2025-05-07T16:41:46Z
#url: https://api.github.com/gists/bf4da265c53047bc4028edc409ad9635
#owner: https://api.github.com/users/birkin

# /// script
# requires-python = "==3.8.*"
# dependencies = ["lxml==4.9.1"]
# ///

"""
Script to test lxml installation on redhat-7 server.

Uses python PEP-723 inline-script-metadata to allow uv to run without venv installs.

Usage: 
$ uv run ./lxml_install_test.py
"""

from lxml import etree


def check_lxml():
    ## show version of lxml installed -------------------------------
    try:
        print("lxml version:", etree.LXML_VERSION)
    except AttributeError:
        print("lxml version not found.")    
    ## Check if lxml is working correctly ---------------------------
    try:
        root = etree.Element("root")
        tree = etree.ElementTree(root)
        print("lxml is working correctly.")
    except Exception as e:
        print(f"lxml is not working correctly: {e}")
    ## Check if lxml can parse XML ----------------------------------
    try:
        xml_string = "<root><child>text</child></root>"
        root = etree.fromstring(xml_string)
        print("lxml can parse XML.")
    except Exception as e:
        print(f"lxml cannot parse XML: {e}")


if __name__ == "__main__":
    check_lxml()