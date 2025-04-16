#date: 2025-04-16T16:52:29Z
#url: https://api.github.com/gists/dcd99688abf94107c398538343d53aed
#owner: https://api.github.com/users/VandanPatel18

import networkx as nx
import matplotlib.pyplot as plt

# Create a sample graph
G = nx.Graph()
edges = [
    ("A", "B", 4),
    ("A", "C", 2),
    ("B", "C", 5),
    ("B", "D", 10),
    ("C", "E", 3),
    ("E", "D", 4),
    ("D", "F", 11),
    ("E", "F", 5),
    ("D", "F", 1),
]
G.add_weighted_edges_from(edges)

# Dijkstra's algorithm using NetworkX for shortest path from 'A'
source_node = "A"
shortest_paths = nx.single_source_dijkstra_path(G, source=source_node)
shortest_distances = nx.single_source_dijkstra_path_length(G, source=source_node)

# Print shortest paths and distances
print("Shortest Paths from node A:")
for target, path in shortest_paths.items():
    print(f"{source_node} -> {target}: {path} (Distance: {shortest_distances[target]})")

# Plotting the graph with edge weights
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Graph with Weighted Edges")
plt.show()