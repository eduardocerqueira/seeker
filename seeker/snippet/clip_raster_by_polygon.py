#date: 2024-01-12T16:59:19Z
#url: https://api.github.com/gists/93df7db1333fe7a102035776f875795b
#owner: https://api.github.com/users/nofurtherinformation

import geopandas as gpd
import rasterio

def clip_raster_by_polygon(
    gdf: gpd.GeoDataFrame, 
    row_number: int, 
    raster_file_path: str,
    output_path: str) -> None:
    """
    Clip a raster to the area of a given polygon

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        Geodataframe to clip features from
    row_number: int
        Row to clip from gdf
    raster_file_path: str
        Input raster image path, expecting a geotif
    output_path: str
        Path to output file
    """

    #     Read in raster to clip
    with rasterio.open(raster_file_path) as src:
        try:
    #             get the low level polygon data
            polygon = gdf.geometry.values[row_number].__geo_interface__
        except Exception as e:
            print(f'[{id}] failed reading', e)
            return
        # Extract the raster data within the polygon
        out_image, out_transform = mask(src, [polygon], crop=True)
        # write updated profile
        profile = src.profile
        profile.update(
            height = out_image.shape[1],
            width = out_image.shape[2],
            transform = out_transform
        )
        try:
            with rasterio.open(output_path, "w", **profile) as dst:
                print(f'[{id}] writing')
                dst.write(out_image)
        except Exception as e:
            print(f'[{id}] failed writing', e)
            return