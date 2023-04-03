#date: 2023-04-03T17:10:13Z
#url: https://api.github.com/gists/5840f929b604006309c7ca6c88708d3b
#owner: https://api.github.com/users/yulleyi

import networkx as nx

def create_major_scale_graph():
    major_scale_pattern = [2, 2, 1, 2, 2, 2, 1]
    chromatic_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Create an empty directed graph
    major_scale_graph = nx.DiGraph()

    # Add nodes for each chromatic note
    for note in chromatic_notes:
        major_scale_graph.add_node(note)

    # Add edges for each whole and half step in the major scale pattern
    for root_note in chromatic_notes:
        root_note_index = chromatic_notes.index(root_note)
        for step in major_scale_pattern:
            root_note_index = (root_note_index + step) % 12
            scale_note = chromatic_notes[root_note_index]

            # Add an edge from the root note to the scale note
            major_scale_graph.add_edge(root_note, scale_note)

    return major_scale_graph

def generate_major_scale_from_graph(major_scale_graph, root_note):
    # Retrieve the major scale notes by following the edges from the root note
    major_scale_notes = list(major_scale_graph.successors(root_note))
    return major_scale_notes

major_scale_graph = create_major_scale_graph()