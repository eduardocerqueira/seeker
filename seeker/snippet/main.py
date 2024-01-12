#date: 2024-01-12T17:06:32Z
#url: https://api.github.com/gists/2f1c012705a7346ed126a81746985c71
#owner: https://api.github.com/users/afletcher53

import matplotlib.pyplot as plt
import networkx as nx

class vertex(object):
    def __init__(self, name,cost=0, calories = 0) -> None:
        self.name: str = name
        self.cost: int = cost
        self.calories: int = calories

    def __str__(self) -> str:
        return str(self.name)

main_menu_items = ["Pizza", "Chips","Burger", "Drink"]
sub_menu_items = ["No Salt", "No Sugar", "No Ice", "No Cheese", "Coke"]

# Create vertices
v0 = vertex("Order")
v1 = vertex("Pizza", 10, 100)
v2 = vertex("Chips", 5, 200)
v3 = vertex("No Salt", 0, 0)


v5 = vertex("No Sugar", 0, 0)
v6 = vertex("No Ice", 0, 0)
v7 = vertex("No Cheese", 0, -100)
v8 = vertex("Burger", 10, 100)
v9 = vertex("Coke", 5, 200)

v10 = vertex("Drink_1", 5, 0)
v11 = vertex("Drink_2", 5, 0)
v12 = vertex("Drink_3", 5, 0)
v13 = vertex("Drink_4", 5, 0)
v14 = vertex("Drink_5", 5, 0)
v15 = vertex("Drink_6", 5, 0)
v16 = vertex("Drink_7", 5, 0)
v17 = vertex("Drink_8", 5, 0)
v18 = vertex("Drink_9", 5, 0)
v19 = vertex("Drink_10", 5, 0)


G = nx.DiGraph()

G.add_node(v0.name, data = v0)
G.add_node(v1.name, data = v1)
G.add_node(v2.name, data = v2)
G.add_node(v3.name, data = v3)


G.add_edge(v0.name, v1.name)
G.add_edge(v0.name, v2.name)
G.add_edge(v2.name, v3.name)

G2 = nx.DiGraph()
G2.add_node(v0.name, data = v0)
G2.add_node(v1.name, data = v1)
G2.add_node(v2.name, data = v2)
G2.add_node(v3.name, data = v3)
G2.add_node(v10.name, data = v10)

G2.add_edge(v0.name, v1.name)
G2.add_edge(v0.name, v2.name)
G2.add_edge(v0.name, v10.name)
G2.add_edge(v10.name, v9.name)
G2.add_edge(v2.name, v3.name)

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(G)
pos2 = nx.spring_layout(G2)

color_map = []
color_map2 = []

for node in G:
    if node in main_menu_items:
        color_map.append('red')
    elif node in sub_menu_items:
        color_map.append('green')
    else: 
        color_map.append('blue')
for node in G2:
    if node in main_menu_items:
        color_map2.append('red')
    elif node in sub_menu_items:
        color_map2.append('green')
    else: 
        color_map2.append('blue')

nx.draw(G, pos, with_labels=True, node_size=2000, node_color=color_map, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels={})
plt.title("Graph Visualization with Vertex Class")
plt.show()

nx.draw(G2, pos2, with_labels=True, node_size=2000, node_color=color_map2, font_size=15)
nx.draw_networkx_edge_labels(G2, pos, edge_labels={})
plt.title("Graph Visualization with Vertex Class")
plt.show()

def calculate_total_calories(graph, start_node):
    total_calories = 0
    visited = set()
    
    def dfs(node):
        if node not in visited:
            visited.add(node)
            total_calories = graph.nodes[node]['data'].calories
            for neighbor in graph.neighbors(node):
                total_calories += dfs(neighbor)
        return total_calories
    
    return dfs(start_node)

total_calories = calculate_total_calories(G, "Order")
print("Total Calories:", total_calories)


def calculate_total_cost(graph, start_node):
    total_cost = 0
    visited = set()
    
    def dfs(node):
        if node not in visited:
            visited.add(node)
            total_cost = graph.nodes[node]['data'].cost
            for neighbor in graph.neighbors(node):
                total_cost += dfs(neighbor)
        return total_cost
    
    return dfs(start_node)

print("Total Cost:", calculate_total_cost(G, "Order"))
print("Similarity Graph Edit Distance:", nx.graph_edit_distance(G, G2))