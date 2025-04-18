#date: 2025-04-18T16:35:53Z
#url: https://api.github.com/gists/4d8fd791ffdae47e822c9cf0b22673bf
#owner: https://api.github.com/users/barronh

__doc__ = """
Grid US Census Demographic Profile
==================================

Produce gridded TIGER Demographic Profile files for a subset of variables in 
the an American Community Survey from either a 1-year or 5-year input files.

Requires:
- python>=3.7
- numpy
- pandas
- xarray
- netcdf4
- shapely
- geopandas
"""
__version__ = '1.0.0'
#%pip install geopandas shapely numpy xarray netcdf4
import geopandas as gpd
from shapely import box
import numpy as np
import os
from functools import reduce


layer = 'ACS_2022_5YR_BG'  # COUNTY or BG or TRACT
gdbpath = f'{layer}.gdb'
url = f'https://www2.census.gov/geo/tiger/TIGER_DP/2022ACS/{gdbpath}.zip'
grid = '0.1x0.1'
ncpath = f'{layer}_{grid}.nc'

if os.path.exists(ncpath):
  raise IOError(f'{ncpath} file exists; remove to remake')

if not os.path.exists(gdbpath):
  !wget -N --no-check-certificate {url}
  !unzip {gdbpath}.zip

# Take a look at the layers:
# print(gpd.list_layers(gdbpath))

# Change the resolution and bounds of the grid here
dx, dy = eval(grid.replace('x', ','))
hdx = dx / 2
hdy = dy / 2
lonc = np.arange(-128, -65 + hdx, dx)
latc = np.arange(22, 52 + hdy, dy)

geom = [
    box(lonc[i] - hdx, latc[j] - hdx, lonc[i] + hdx, latc[j] + hdx)
    for j, i in np.ndindex(latc.size, lonc.size)
]
data = [
    dict(lon=lonc[i], lat=latc[j])
    for j, i in np.ndindex(latc.size, lonc.size)
]
griddf = gpd.GeoDataFrame(
    data, geometry=geom, crs=4326
).set_index(['lat', 'lon'])


gdf = gpd.read_file(gdbpath, layer=layer).to_crs(griddf.crs)

# https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/documentation/ACS20225YR_Table_Shells.txt
lt35male = [f'B01001_E{i:03d}' for i in range(3, 13)]
lt35female = [f'B01001_E{i:03d}' for i in range(27, 37)]
ge35male = [f'B01001_E{i:03d}' for i in range(13, 26)]
ge35female = [f'B01001_E{i:03d}' for i in range(37, 50)]
askeys = {
    'Pop': ['B01001_E001'],
    'Pop_LT35y': lt35male + lt35female,
    'Pop_GE35y': ge35male + ge35female,
}
rakeys = {
    'Pop_White': ['B02001_E002'],
    'Pop_Black': ['B02001_E003'],
    'Pop_Native': ['B02001_E004'],
    'Pop_Asian': ['B02001_E005'],
    'Pop_Pacif': ['B02001_E006'],
    'Pop_Other': ['B02001_E007'],
    'Pop_Multi': ['B02001_E008'],
}

ascols = ['GEOIDFQ'] + reduce(list.__add__, askeys.values())
asdf = gpd.read_file(gdbpath, layer='X01_AGE_AND_SEX', columns=ascols)

racols = ['GEOIDFQ'] + reduce(list.__add__, rakeys.values())
radf = gpd.read_file(gdbpath, layer='X02_RACE', columns=racols)

# ordered by alignment with gdf
gasdf = asdf.set_index('GEOIDFQ').loc[gdf.GEOIDFQ]
gradf = radf.set_index('GEOIDFQ').loc[gdf.GEOIDFQ]

# Add summary variables as the sum of others
for ok, iks in askeys.items():
  gdf[ok] = gasdf[iks].sum(axis=1).values
for ok, iks in rakeys.items():
  gdf[ok] = gradf[iks].sum(axis=1).values

popkeys = [k for k in gdf.columns if k.startswith('Pop')]
gdf['source_area'] = gdf.area
ol = gpd.overlay(griddf.reset_index(), gdf[['geometry', 'source_area'] + popkeys])
olw = ol.copy()
olw[popkeys] = ol[popkeys].multiply(ol.area / ol['source_area'], axis=0)
olw['intx_area'] = ol.area
gridsumdf = olw.groupby(['lat', 'lon'])[popkeys].sum()
gridsumdf = griddf.join(gridsumdf)
ds = gridsumdf.to_xarray()
for k, v in ds.data_vars.items():
  v.encoding.update(zlib=True, complevel=1)
  v.attrs.update(long_name=k, units='1')


for ok, iks in askeys.items():
  ds[ok].attrs['description'] = 'ACS Code: ' + ', '.join(iks)
for ok, iks in rakeys.items():
  ds[ok].attrs['description'] = 'ACS Code: ' + ', '.join(iks)

refs = url
comment = f'Assigned based on area-overlap using geopandas v{gpd.__version__}'
title = f'{gdbpath} gridded to {dx} by {dy} degrees'
ds.attrs.update(
    title=title, institution='US EPA', author='Henderson',
    references=refs, comment=comment,
)
ds[popkeys].to_netcdf(ncpath, format='NETCDF4_CLASSIC')