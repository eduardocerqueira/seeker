#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# city_map/map_loader.py

import osmnx as ox

def load_city_graph(filepath="tabriz_iran.graphml"):
    print(f"📂 Loading city graph from {filepath}...")
    G = ox.load_graphml(filepath)
    print("✅ Graph loaded.")
    return G
