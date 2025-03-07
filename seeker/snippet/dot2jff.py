#date: 2025-03-07T16:49:40Z
#url: https://api.github.com/gists/47dd83df39961f009c05a330ae10d7f5
#owner: https://api.github.com/users/Hurka5

#!/bin/python3
import sys
import re
import pygraphviz as pgv
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Check if the script is being run with a file argument
if len(sys.argv) < 2:
    print("Usage: python dot2jff.py <path_to_dot_file>")
    sys.exit(1)

# The first argument after the script name is the path to the .dot file
dot_file = sys.argv[1]

# Load the Graphviz DOT file using pygraphviz
graph = pgv.AGraph(dot_file)

# Create the root element of the XML
root = ET.Element("structure")

# Set type
type_node = nodes_element = ET.SubElement(root, "type")
type_node.text = "fa"

# Create elements for nodes
node_lookup_table = {}
automaton_nodes = ET.SubElement(root, "automaton")
for node in graph.nodes():
    #print(node)

    if node == "accepting" or node == "start":
        continue

    # state
    node_element = ET.SubElement(automaton_nodes, "state")
    id = graph.nodes().index(node)
    node_element.set("id", str(id))
    node_element.set("name", node)

    node_lookup_table[node] = str(id)

    # x
    x_element = ET.SubElement(node_element, "x")
    x_element.text = str(id*50.0)

    # y
    y_element = ET.SubElement(node_element, "y")
    y_element.text = str(id*50.0)


    for edge in graph.edges():
        # initial
        if edge[0] == "start" and edge[1] == node:
            ET.SubElement(node_element, "initial")
        # final
        if edge[1] == "accepting" and edge[0] == node:
            ET.SubElement(node_element, "final")



# Create elements for edges
for edge in graph.edges():
    if edge[0] == "start":
        continue
    if edge[1] == "accepting":
        continue
    edge_element = ET.SubElement(automaton_nodes, "transition")

    # from
    from_element = ET.SubElement(edge_element, "from")
    from_element.text = node_lookup_table[edge[0]]

    # to
    to_element = ET.SubElement(edge_element, "to")
    to_element.text = node_lookup_table[edge[1]]

    # read
    read_element = ET.SubElement(edge_element, "read")
    read_element.text = graph.get_edge(edge[0], edge[1]).attr.get('label', None)


# Convert the XML tree to a string
xml_str = ET.tostring(root, encoding="utf-8").decode()

# Use minidom to prettify
pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ",encoding="UTF-8",standalone=False)

# Combine declaration, comment, and XML string
final_xml = pretty_xml.decode("utf-8")

# Write to output file
f = open("out.jff", "w")
f.write(final_xml)
f.close()
print(final_xml)
