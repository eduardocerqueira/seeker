#date: 2025-08-25T17:08:20Z
#url: https://api.github.com/gists/1010989d4e976fe26d1a5d4b2c533cff
#owner: https://api.github.com/users/ayhanasghari

# city_map/building_loader.py

import geopandas as gpd

def load_buildings(filepath="tabriz_iran_buildings.gpkg"):
    print(f"📂 Loading buildings from {filepath}...")
    gdf = gpd.read_file(filepath)
    buildings = gdf[gdf.geometry.type == 'Polygon']
    print(f"✅ {len(buildings)} buildings loaded.")
    return buildings
