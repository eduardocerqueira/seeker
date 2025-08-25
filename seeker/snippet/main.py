#date: 2025-08-25T17:01:57Z
#url: https://api.github.com/gists/35dd1a68313395a2f03ce6853c984081
#owner: https://api.github.com/users/ayhanasghari

# 

from city_map.city_map import CityMap

def main():
 #  city = CityMap(    graph_path="tabriz_iran.graphml",    buildings_path="tabriz_iran_building.gpkg")

    city = CityMap()
    city.load_graph()
    city.load_buildings()
    city.compute_building_centers()
    city.plot_map()

if __name__ == "__main__":
    main()
