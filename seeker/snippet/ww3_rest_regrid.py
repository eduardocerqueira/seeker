#date: 2025-05-19T16:44:24Z
#url: https://api.github.com/gists/a3322224f10c67170e1e87e10bb618ef
#owner: https://api.github.com/users/AliS-Noaa

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ali.Salimi-Tarazouj
The script will regrid WW3 restart NetCDF file using ESMPy.
It requires the scrip.nc file for both source and destination grid.
USEAGE:
    mpirun -n 1 python ww3_rest_intp.py --restart_file 20241125.060000.restart.ww3.nc --src_scrip scrip.uglo_15km.nc --dst_scrip hafswav_a.SCRIP.nc --output_file restar_interp.nc
"""

import esmpy
import numpy as np
import netCDF4 as nc
import os
import argparse
import time
import logging

start_time = time.time()

#######################################
#     Parse command-line arguments    #
#######################################
parser = argparse.ArgumentParser(description="Regrid WW3 restart NetCDF file using ESMPy.")
parser.add_argument("--restart_file", required=True, help="Path to input restart NetCDF file.")
parser.add_argument("--src_scrip", required=True, help="Path to source SCRIP mesh file.")
parser.add_argument("--dst_scrip", required=True, help="Path to destination SCRIP grid file.")
parser.add_argument("--output_file", required=True, help="Path to output NetCDF file.")
parser.add_argument("--log_file", default="regrid.log", help="Path to log file (default: regrid.log)")
args = parser.parse_args()

######################################
#         Setup logging              #
######################################
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logging.info("Starting WW3 regridding script")

#####################################
#         File paths                #
#####################################
restart_file = args.restart_file
src_scrip = args.src_scrip
dst_scrip = args.dst_scrip
output_file = args.output_file

####################################
#     Get variables to regrid      #
####################################
src_nc = nc.Dataset(restart_file, "r")
vars_to_regrid = []
for v in src_nc.variables:
    var = src_nc.variables[v]
    if ("time" in var.dimensions and "nx" in var.dimensions) and var.ndim in (2, 3):
        if var.ndim == 3:
            ny_dim = var.shape[var.dimensions.index("ny")]
            if ny_dim != 1:
                continue  # skip variables where ny ▒~I|  1
        vars_to_regrid.append(v)

#####################################
#       Open input data             #
#####################################
time_val = src_nc.variables["time"][:]
nt = len(time_val)
nx_src = src_nc.dimensions["nx"].size
ny_src = src_nc.dimensions["ny"].size if "ny" in src_nc.dimensions else 1

# Scalars
nk = src_nc.variables["nk"][:].item()
nth = src_nc.variables["nth"][:].item()

# Destination grid mask
with nc.Dataset(dst_scrip) as dst_nc:
    dst_mask = dst_nc.variables["grid_imask"][:]
    nx_dst = dst_nc.dimensions["grid_size"].size

####################################
#        Initialize ESMF           #
####################################
esmpy.Manager()

# Build mesh/grid
src_mesh = esmpy.Mesh(filename=src_scrip, filetype=esmpy.FileFormat.SCRIP)
dst_grid = esmpy.Grid(filename=dst_scrip, filetype=esmpy.FileFormat.SCRIP)

# Build fields with
src_field = esmpy.Field(src_mesh, meshloc=esmpy.MeshLoc.ELEMENT, ndbounds=[nt])
dst_field = esmpy.Field(dst_grid, staggerloc=esmpy.StaggerLoc.CENTER, ndbounds=[nt])

# Setup regridder
regrid = esmpy.Regrid(
    srcfield=src_field,
    dstfield=dst_field,
    regrid_method=esmpy.RegridMethod.BILINEAR,
    unmapped_action=esmpy.UnmappedAction.IGNORE,
    dst_mask_values=[0]
)

# Remove old output file
if os.path.exists(output_file):
    os.remove(output_file)
    logging.info(f"Removed old output file: {output_file}")

###############################
#    Create output NetCDF     #
###############################
with nc.Dataset(output_file, "w") as out_nc:
    out_nc.createDimension("time", nt)
    out_nc.createDimension("ny", ny_src)
    out_nc.createDimension("nx", nx_dst)

    # Time
    time_var = out_nc.createVariable("time", "f8", ("time",))
    time_var[:] = time_val
    for attr in src_nc.variables["time"].ncattrs():
        setattr(time_var, attr, getattr(src_nc.variables["time"], attr))

    # Scalars
    nk_var = out_nc.createVariable("nk", "i4")
    nk_var[...] = nk
    nk_var.long_name = "number of frequencies"

    nth_var = out_nc.createVariable("nth", "i4")
    nth_var[...] = nth
    nth_var.long_name = "number of direction bins"

    # Loop through variables
    for varname in vars_to_regrid:
        logging.info(f"Regridding: {varname}")
        src_data = src_nc.variables[varname][:]  # shape: (time, ny, nx) or (time, nx)

        # Squeeze to (time, nx)
        src_data_2d = src_data.reshape(nt, nx_src)

        # Regrid
        src_field.data[...] = np.asfortranarray(src_data_2d.T)
        dst_field = regrid(src_field, dst_field)
        dst_data = dst_field.data[...].T  # shape (time, nx_dst)

        # Write output
        out_var = out_nc.createVariable(varname, "f4", ("time", "ny", "nx"), fill_value=9.96921e36)
        out_var[:, 0, :] = dst_data

        for attr in src_nc.variables[varname].ncattrs():
            if attr != "_FillValue":
                setattr(out_var, attr, getattr(src_nc.variables[varname], attr))

# === Done ===
elapsed = time.time() - start_time
logging.info(f" Done. Output written to: {output_file}")
logging.info(f" Elapsed time: {elapsed:.2f} seconds")
