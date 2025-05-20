#date: 2025-05-20T17:13:26Z
#url: https://api.github.com/gists/3c9e3d7bb1db5be9a09a3a6d6d26ebee
#owner: https://api.github.com/users/dgketchum

import os
import csv
import requests
import time
import geopandas
from shapely.geometry import Point

"""
Montana State Library: Montana's Tallest Peaks by Mountain Range
https://msl.mt.gov/geoinfo/geography/geography_facts/montanaxs_tallest_peaks_by_mountain_range

Pulling most of these by GNIS name lookup and filling in the missing values under MISSING
by finding the peak and getting coordinates from Google Earth.
"""

WATERDATA_API_URL = "https://dashboard.waterdata.usgs.gov/service/geocoder/get/location/1.0"
STATE_ABBREVIATION = "MT"
REQUEST_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 2.0

MISSING = {'Bears Paw Mountains': (48.148593, -109.65075),

           # Beartooth Mountains GNIS lookup gets first result from Tobacco Root
           'Beartooth Mountains': (45.1841, -109.79129),

           # exact point of Beaverhead Mtns HP is in Idaho
           'Beaverhead Mountains': (44.447227, -112.996476),

           # Bighorn's Montana HP is drive-up
           # 'Bighorn Mountains': (45.00098, -107.91139),
           # Bighorn Overall HP is Cloud Peak
           'Bighorn Mountains': (44.382147, -107.173951),

           'Blacktail Mountains': (48.01077, -114.36367),
           'Chalk Buttes': (45.708559, -104.734215),
           'Henrys Lake Mountains': (44.7633, -111.3906),
           'John Long Mountains': (46.437778, -113.447778),
           'Little Snowy Mountains': (46.752222, -109.173333),
           'Long Pines': (45.639974, -104.184526),
           'North Moccasin Mountains': (47.31448, -109.46758),
           'Ruby Range': (45.312778, -112.228056),
           'South Moccasin Mountains': (47.1725, -109.523889),
           'Wolf Mountains': (45.03603, -107.19281)}


def get_coordinates_from_gnis(mountain_name, range_):
    params = {
        "term": mountain_name,
        "include": "gnis",
        "states": STATE_ABBREVIATION,
        "maxSuggestions": 1
    }
    headers = {}
    response = requests.get(WATERDATA_API_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    if data and len(data) > 0:
        location = data[0]
        if location.get("Source") == "gnis" and "Latitude" in location and "Longitude" in location:
            print(f'{mountain_name}: {location["Latitude"]:.2f}, {location["Longitude"]:.2f}')
            return location["Latitude"], location["Longitude"]
    else:
        raise ValueError

def process_mountains_csv(input_csv_filepath, output_csv_filepath, output_shapefile_filepath):
    processed_mountains_for_csv = []
    mountains_for_geodataframe = []

    with open(input_csv_filepath, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)

        for row_number, row_dict in enumerate(reader, 1):
            name_from_csv = row_dict.get('Name')
            climbed = row_dict.get('done')
            range = row_dict.get('range')
            elev_ft = row_dict.get('elev_ft')
            status_msg = ""
            lat, lon = None, None

            current_attributes = dict(row_dict)

            if name_from_csv and name_from_csv.strip():
                stripped_name = name_from_csv.strip()
                current_attributes['Name'] = stripped_name

                if range in MISSING:
                    loc = MISSING[range]
                    print(f'{stripped_name}: {loc[0]:.2f}, {loc[1]:.2f}')
                else:
                    loc = get_coordinates_from_gnis(stripped_name, range)

                lat, lon = loc[0], loc[1]
                time.sleep(DELAY_BETWEEN_REQUESTS)

                status_msg = "Success"
                point_data = current_attributes.copy()

                point_data['geometry'] = Point(float(lon), float(lat))
                point_data['latitude'] = float(lat)
                point_data['longitude'] = float(lon)

                point_data['status'] = status_msg
                mountains_for_geodataframe.append(point_data)

            else:
                status_msg = "Skipped - Empty name in CSV"
                current_attributes['Name'] = name_from_csv if name_from_csv else ''

            csv_entry = {
                'name': current_attributes['Name'],
                'latitude': lat if lat is not None else 'N/A',
                'longitude': lon if lon is not None else 'N/A',
                'range': range,
                'elev_ft': elev_ft,
                'done': climbed
            }
            processed_mountains_for_csv.append(csv_entry)

    if processed_mountains_for_csv:
        with open(output_csv_filepath, mode='w', newline='', encoding='utf-8') as outfile:
            fieldnames = list(csv_entry.keys())
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_mountains_for_csv)

    if mountains_for_geodataframe:
        gdf = geopandas.GeoDataFrame(mountains_for_geodataframe, crs="EPSG:4326")

        sanitized_columns = {}

        for col_name in gdf.columns:
            if col_name == 'geometry':
                sanitized_columns[col_name] = 'geometry'
                continue

            s_name = str(col_name).replace(' ', '_').replace('.', '_').replace('-', '_')
            s_name = s_name[:10]

            original_s_name = s_name
            count = 1
            while s_name in sanitized_columns.values():
                s_name = original_s_name[:10 - len(str(count))] + str(count)
                count += 1
            sanitized_columns[col_name] = s_name

        gdf.rename(columns=sanitized_columns, inplace=True)
        gdf.to_file(output_shapefile_filepath, driver='ESRI Shapefile')


if __name__ == "__main__":

    d = os.path.expanduser('~')

    input_file = os.path.join(d, 'mt_highpoints_noCoords.csv')
    output_csv = os.path.join(d, 'mt_highpoints.csv')
    output_shp = os.path.join(d, 'mt_highpoints.shp')

    process_mountains_csv(input_file, output_csv, output_shp)

# ========================= EOF ====================================================================
