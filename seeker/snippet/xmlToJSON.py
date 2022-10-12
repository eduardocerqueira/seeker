#date: 2022-10-12T17:35:31Z
#url: https://api.github.com/gists/ab640afe860d0e6875c8d24c03e9a291
#owner: https://api.github.com/users/tjbearse

# converts a basic xml blockly block to json (does not migrate mutators)

import xml.etree.ElementTree as ET
import json
import sys


"""
example block
<block type="sheets_formula" id="root">
        <value name="FORMULA">
                <block type="sheets_ABS" id="root">
                        <value name="ARG0">
                                <block type="sheets_number" id="2">
                                        <field name="NUM">2</field>
                                </block>
                        </value>
                </block>
        </value>
</block>
"""

def main():
    tree = ET.parse(sys.stdin)
    root = tree.getroot()
    b = extractBlock(root)
    print(json.dumps(b, indent=4))

def extractBlock(node: ET.Element):
    b = dict(
        type=node.attrib['type']
    )
    fields=extractFields(node)
    if fields:
        b['fields'] = fields

    inputs = extractInputs(node)
    if inputs:
        b['inputs'] = inputs
    return b

def extractFields(node: ET.Element):
    return {
        child.attrib['name']: child.text
        for child in node if child.tag == 'field'
    }

def extractInputs(node: ET.Element):
    return {
        child.attrib['name']:
            dict(block=extractBlock(child[0])) if len(child) == 1 else None
        for child in node if child.tag == 'value'
    }

if __name__ == '__main__':
    main();
