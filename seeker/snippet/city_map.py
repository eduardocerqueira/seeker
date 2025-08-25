#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# city_map/city_map.py

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import osmnx as ox

from city_map.map_loader import load_city_graph
from city_map.building_loader import load_buildings

class CityMap:
    def __init__(self, graph_path="tabriz_iran.graphml", buildings_path="tabriz_iran_buildings.gpkg"):
        self.graph_path = graph_path
        self.buildings_path = buildings_path
        self.graph = None
        self.buildings = None
        self.building_centers = []

    def load_graph(self):
        self.graph = load_city_graph(self.graph_path)

    def load_buildings(self):
        self.buildings = load_buildings(self.buildings_path)

    def compute_building_centers(self):
        print("📌 Computing building centers...")
        self.building_centers = []
        for geom in self.buildings.geometry:
            if isinstance(geom, Polygon):
                self.building_centers.append(geom.centroid)
        print(f"✅ {len(self.building_centers)} centers computed.")

    def plot_map(self):
       
        print("🛣️ Plotting street graph only...")

        if self.graph is None:
            print("⚠️ Graph not loaded. Call load_graph() first.")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        ox.plot_graph(
            self.graph,
            ax=ax,
            show=False,
            close=False,
            node_size=0,
            edge_color="black",
            edge_linewidth=0.6
        )

        ax.set_title(f"Street Graph of {self.graph_path}", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()


