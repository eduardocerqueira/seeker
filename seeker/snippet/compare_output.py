#date: 2026-02-18T17:39:11Z
#url: https://api.github.com/gists/24575538813f66887ed7cbfcaa36903d
#owner: https://api.github.com/users/czarakas

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

var='pr'
v1_pr=xr.open_zarr("s3://carbonplan-scratch/srm-scratch/v0.3_SouthAfrica/"+var+"_data.zarr/scenario_debiased_downscaled/")
v2_pr =xr.open_zarr("s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM_"+var+"_000_global_SSP245.zarr/")

var='tas'
v1_tas=xr.open_zarr("s3://carbonplan-scratch/srm-scratch/v0.3_SouthAfrica/"+var+"_data.zarr/scenario_debiased_downscaled/")
v2_tas =xr.open_zarr("s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM_"+var+"_000_global_SSP245.zarr/")

xmin = v1_tas.lon.min().item() 
xmax = v1_tas.lon.max().item() 
ymin = v1_tas.lat.min().item() 
ymax = v1_tas.lat.max().item() 
v2_pr_subset = v2_pr.sel( lon=slice(xmin, xmax), lat=slice(ymin, ymax))
v2_tas_subset = v2_tas.sel( lon=slice(xmin, xmax), lat=slice(ymin, ymax))

v2_tas_subset_mean = v2_tas_subset.mean(dim='time').compute()
v2_pr_subset_mean = v2_pr_subset.mean(dim='time').compute()

v1_tas_mean = v1_tas.mean(dim='time').compute()
v1_pr_mean = v1_pr.mean(dim='time').compute()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
((v2_tas_subset_mean["tas"]-v1_tas_mean["tas"])).plot(cmap=plt.cm.coolwarm, vmin=-0.4, vmax=0.4)
plt.title("$\Delta$ Mean temperature (global version - regional version)")
plt.subplot(1,2,2)
((v2_pr_subset_mean["pr"]-v1_pr_mean["pr"])).plot()
plt.title("$\Delta$ Mean precipitation (global version - regional version)")
plt.tight_layout()