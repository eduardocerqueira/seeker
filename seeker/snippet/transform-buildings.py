#date: 2022-02-09T17:10:27Z
#url: https://api.github.com/gists/7e1f0e653288cae51aa8c6411329e7f9
#owner: https://api.github.com/users/JEPooleyOS

# Move buildings according to their area
WIDTH = 2405
SPACE = 5

polygons = []
shift_x, shift_y = 0, 0
range_y, range_x = 0, 0
for building in local_buildings_gdf.itertuples():

    # Extract geometry and bounding box
    geometry = building.geometry
    min_x, min_y, max_x, max_y = geometry.bounds
    range_x = max_x - min_x

    # Check whether to wrap to next line
    if shift_x + range_x > WIDTH:
        shift_y += range_y + SPACE
        shift_x, range_y = 0, 0

    # Translate geometry
    shifted_geometry = translate(geometry,
                                 xoff=shift_x - min_x,
                                 yoff=shift_y - min_y)

    # Update polygons list
    polygons.append(shifted_geometry)

    # Update shift parameters
    range_y = max(max_y - min_y, range_y)
    shift_x += max_x - min_x + SPACE
   
# Create GeoSeries
gs = gpd.GeoSeries(polygons)