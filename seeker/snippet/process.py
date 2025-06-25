#date: 2025-06-25T16:57:54Z
#url: https://api.github.com/gists/88d2353db8ba9c84ae0e290e3006e254
#owner: https://api.github.com/users/uturuncoglu

import os
import numpy as np
import xarray as xr
import uxarray as ux
from uxarray.conventions import ugrid
import conduit
from conduit import Node

import holoviews as hv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews.feature as gf
import panel as pn

data_dir = '/glade/work/turuncu/COP/casper_fresh/data'

my_channel = Node()
my_channel.load(os.path.join(data_dir, 'my_channel'))

ds_grid = xr.Dataset()
ds_grid["node_lon"] = xr.DataArray(
        my_channel['data/coords/values/node_lon'], 
        dims=ugrid.NODE_DIM, 
        attrs=ugrid.NODE_LON_ATTRS
    )
ds_grid["node_lat"] = xr.DataArray(
        my_channel['data/coords/values/node_lat'],
        dims=ugrid.NODE_DIM,
        attrs=ugrid.NODE_LAT_ATTRS
    )
ds_grid["face_lon"] = xr.DataArray(
        my_channel['data/coords/values/face_lon'], 
        dims=ugrid.FACE_DIM, 
        attrs=ugrid.FACE_LON_ATTRS
    )
ds_grid["face_lat"] = xr.DataArray(
        my_channel['data/coords/values/face_lat'],
        dims=ugrid.FACE_DIM,
        attrs=ugrid.FACE_LAT_ATTRS
    )
n_face =  my_channel['data/dimension/n_face']
n_max_face_nodes =  my_channel['data/dimension/n_max_face_nodes']
ds_grid["face_node_connectivity"] = xr.DataArray(
        my_channel['data/topologies/mesh/elements/face_node_connectivity'].reshape((n_face,n_max_face_nodes)),
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

ds_data = xr.Dataset()
ds_data["mask"] = xr.DataArray(
        my_channel['data/mask/values/face_mask'],
        dims=ugrid.FACE_DIM
)

uxgrid = ux.Grid(ds_grid, source_grid_spec="UGRID")
uxgrid.validate()

uxds = ux.UxDataset(uxgrid=uxgrid)
uxds_new = uxds.assign(ds_data)

plot_opts = {"width": 700, "height": 350}
hv.extension("bokeh")

features = gf.coastline(
    projection=ccrs.PlateCarree(), line_width=1, scale="110m"
)

plot1 = uxds_new['mask'].plot(
    rasterize=True,
    periodic_elements="exclude",
    title="Wind",
    clabel="V [m s-1]",
    cmap="plasma",
    **plot_opts,
) * features

plot2 = uxds_new['mask'].plot(
    rasterize=True,
    periodic_elements="exclude",
    title="Wind",
    clabel="V [m s-1]",
    cmap="plasma",
    **plot_opts,
) * features

# one row example
panel = pn.Row(plot1, plot2)

# two row example
panel = pn.Column(
    pn.Row(plot1, plot2),
    pn.Row(plot1, plot2)
)

hv.save(plot, "plot.html", backend="bokeh")