#date: 2025-02-28T16:55:03Z
#url: https://api.github.com/gists/9075bab25f9d3eb9c68ba9ea5c0deaa7
#owner: https://api.github.com/users/danielgottbehuet

import os
import math
import requests

# OpenStreetMap Tile-Server URL (kann angepasst werden)
TILE_SERVER_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Definiere die gewünschte Bounding Box für Köln
BOUNDING_BOX = {
    "minLat": 50.8303,  # Südlichster Punkt von Köln
    "maxLat": 51.0846,  # Nördlichster Punkt von Köln
    "minLon": 6.7724,   # Westlichster Punkt von Köln
    "maxLon": 7.1620    # Östlichster Punkt von Köln
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0"
}

# Definiere die Zoomstufen, die geladen werden sollen
ZOOM_LEVELS = [10, 11, 12, 13, 14, 15, 16]

# Umrechnung von Latitude/Longitude in Tile-Koordinaten
def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

# Funktion zum Herunterladen eines Tiles
def download_tile(z, x, y):
    url = TILE_SERVER_URL.format(z=z, x=x, y=y)
    tile_dir = os.path.join("tiles", str(z), str(x))
    tile_path = os.path.join(tile_dir, f"{y}.png")

    if os.path.exists(tile_path):
        print(f"Tile {z}/{x}/{y} existiert bereits.")
        return

    try:
        response = requests.get(url, headers=HEADERS, stream=True)
        response.raise_for_status()
        os.makedirs(tile_dir, exist_ok=True)
        with open(tile_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Gespeichert: {tile_path}")
    except requests.RequestException as e:
        print(f"Fehler beim Laden von {url}: {e}")

# Tiles für eine bestimmte Zoomstufe herunterladen
def download_tiles_for_zoom(z):
    min_x, min_y = lat_lon_to_tile(BOUNDING_BOX["maxLat"], BOUNDING_BOX["minLon"], z)
    max_x, max_y = lat_lon_to_tile(BOUNDING_BOX["minLat"], BOUNDING_BOX["maxLon"], z)

    print(f"Lade Tiles für Zoomstufe {z} ({min_x},{min_y} bis {max_x},{max_y})...")

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            download_tile(z, x, y)

# Hauptfunktion
def main():
    for z in ZOOM_LEVELS:
        download_tiles_for_zoom(z)
    print("Alle Tiles wurden heruntergeladen!")

if __name__ == "__main__":
    main()
