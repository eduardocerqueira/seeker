#date: 2025-08-15T16:56:34Z
#url: https://api.github.com/gists/47f9d52e15d761157913da6e32fdb779
#owner: https://api.github.com/users/morehavoc

#!/usr/bin/env python3
# US County Population Choropleth (Census data)
# - Downloads county boundaries (Cartographic 1:5m) and 2023 county population estimates
# - Produces a matplotlib choropleth with dark red = highly populated, white = sparse
#
# Requirements (the script will attempt to pip install if missing):
#   pandas, geopandas, matplotlib, requests, pyogrio
#
# Usage:
#   python us_county_population_map_from_census.py
#
import os, sys, io, zipfile, subprocess, shutil
from pathlib import Path

def ensure_packages(pkgs):
    import importlib
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

ensure_packages(["pandas", "geopandas", "matplotlib", "requests", "pyogrio"])

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import requests

DATA_DIR = Path("data_us_counties")
DATA_DIR.mkdir(exist_ok=True)
OUT_PNG = Path("us_county_population_map.png")

# Sources
SHAPE_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip"
POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv"

def download(url, dest):
    if not dest.exists():
        print(f"Downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)

# Download data
shape_zip = DATA_DIR / "cb_2018_us_county_5m.zip"
pop_csv = DATA_DIR / "co-est2023-alldata.csv"
download(SHAPE_URL, shape_zip)
download(POP_URL, pop_csv)

# Extract shapefile
shape_dir = DATA_DIR / "shp"
if not shape_dir.exists():
    shape_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(shape_zip, "r") as zf:
        zf.extractall(shape_dir)

# Read shapefile
shp_files = list(shape_dir.glob("*.shp"))
if not shp_files:
    print("Shapefile not found after extraction.")
    sys.exit(1)

gdf = gpd.read_file(shp_files[0])  # has GEOID = 5-digit FIPS
# Read population CSV
df = pd.read_csv(pop_csv, encoding="latin-1")

# Build 5-digit FIPS in CSV
# CSV columns typically include: STATE, COUNTY, STNAME, CTYNAME, POPESTIMATE2023, etc.
for col in ("STATE", "COUNTY"):
    if col not in df.columns:
        print(f"Expected column {col} not found in population CSV.")
        sys.exit(1)

df["STATE"] = df["STATE"].astype(int)
df["COUNTY"] = df["COUNTY"].astype(int)
df["FIPS"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(str).str.zfill(3)

# Choose population column (prefer 2023 estimate)
pop_col = None
for c in ["POPESTIMATE2023", "POPESTIMATE2022", "POPESTIMATE2021", "POPESTIMATE2020"]:
    if c in df.columns:
        pop_col = c
        break
if pop_col is None:
    print("Could not find POPESTIMATE20XX column in CSV.")
    sys.exit(1)

# Filter to counties (SUMLEV==50 is counties)
if "SUMLEV" in df.columns:
    df = df[df["SUMLEV"] == 50].copy()

df_pop = df[["FIPS", pop_col]].rename(columns={pop_col: "population"})

# Merge to geodata
gdf = gdf.merge(df_pop, left_on="GEOID", right_on="FIPS", how="left")

# Some counties may be missing; fill with NaN then drop for plotting color
gdf["population"] = gdf["population"].astype(float)

# Plot
fig = plt.figure(figsize=(14, 9), dpi=150)
ax = fig.add_subplot(111)

cmap = get_cmap("Reds")  # white (low) -> dark red (high)
vals = gdf["population"].values
norm = Normalize(vmin=vals.min(), vmax=vals.max())

gdf.plot(ax=ax, column="population", cmap=cmap, linewidth=0.05, edgecolor="none")

ax.set_axis_off()
ax.set_title("U.S. County Population (Census 2023 estimates)", fontsize=14, pad=14)

# Colorbar
import numpy as np
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02)
cbar.set_label("Population (absolute)", rotation=90)

fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
print(f"Saved: {OUT_PNG.resolve()}")
